import os
import textwrap
import unittest
from tempfile import TemporaryDirectory

from umlfri2.application.importers.java import (
    ImportContext,
    JavaSourceParser,
    JavaEnumConstant,
    JavaTypeModel,
    JavaTypeResolver,
    TypeDescriptor,
)


class JavaImporterParserTests(unittest.TestCase):
    def test_parser_extracts_class_details(self):
        source = textwrap.dedent(
            """
            package com.example;
            import java.util.List;
            public class Foo extends Base implements Runnable {
                private String value;
                protected static List<Bar> bars;

                public Foo(String value) {}

                @Override
                public void run() {}
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "Foo.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        self.assertIn("com.example.Foo", result.types)
        foo = result.types["com.example.Foo"]
        self.assertEqual(foo.extends[0].name, "Base")
        self.assertEqual(foo.implements[0].name, "Runnable")
        self.assertEqual(len(foo.fields), 2)
        self.assertEqual(foo.fields[0].name, "value")
        self.assertEqual(foo.fields[0].type_descriptor.name, "String")
        self.assertEqual(foo.fields[1].type_descriptor.name, "List")
        self.assertEqual(foo.fields[1].type_descriptor.arguments[0].name, "Bar")
        method_names = {method.name for method in foo.methods}
        self.assertIn("Foo", method_names)
        self.assertIn("run", method_names)

    def test_parser_handles_basic_types(self):
        source = textwrap.dedent(
            """
            public class WithPrimitives {
                private int counter;
                protected double[] measurements;
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "WithPrimitives.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["WithPrimitives"]
        primitive_field = next(field for field in model.fields if field.name == "counter")
        array_field = next(field for field in model.fields if field.name == "measurements")
        self.assertEqual(primitive_field.type_descriptor.name, "int")
        self.assertEqual(array_field.type_descriptor.name, "double")
        self.assertEqual(array_field.type_descriptor.dimensions, 1)

    def test_resolver_prefers_imports(self):
        target_primary = JavaTypeModel(
            name="Target",
            package="com.sample",
            kind="class",
            modifiers=set(),
            fields=[],
            methods=[],
            extends=[],
            implements=[],
            imports=ImportContext(),
            source_path="/tmp/Target.java",
        )
        target_secondary = JavaTypeModel(
            name="Target",
            package="org.alt",
            kind="class",
            modifiers=set(),
            fields=[],
            methods=[],
            extends=[],
            implements=[],
            imports=ImportContext(),
            source_path="/tmp/Target2.java",
        )
        source_model = JavaTypeModel(
            name="Source",
            package="com.example",
            kind="class",
            modifiers=set(),
            fields=[],
            methods=[],
            extends=[],
            implements=[],
            imports=ImportContext(direct_imports={"Target": "com.sample.Target"}),
            source_path="/tmp/Source.java",
        )

        models = {
            target_primary.full_name: target_primary,
            target_secondary.full_name: target_secondary,
            source_model.full_name: source_model,
        }

        resolver = JavaTypeResolver(models)
        descriptor = TypeDescriptor(name="Target")
        resolved = resolver.resolve(descriptor, source_model)
        self.assertEqual(resolved, "com.sample.Target")

    def test_parser_adds_default_constructor(self):
        source = textwrap.dedent(
            """
            public class NoConstructor {
                private int value;
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "NoConstructor.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["NoConstructor"]
        constructors = [m for m in model.methods if m.is_constructor]
        self.assertEqual(len(constructors), 1)
        self.assertEqual(constructors[0].name, "NoConstructor")
        self.assertEqual(constructors[0].parameters, [])
        self.assertIn("public", constructors[0].modifiers)

    def test_parser_adds_default_constructor_package_private(self):
        source = textwrap.dedent(
            """
            class PackagePrivate {
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "PackagePrivate.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["PackagePrivate"]
        constructors = [m for m in model.methods if m.is_constructor]
        self.assertEqual(len(constructors), 1)
        self.assertEqual(constructors[0].name, "PackagePrivate")
        self.assertFalse(any(m in constructors[0].modifiers for m in ["public", "protected", "private"]))

    def test_parser_extracts_enum_constants(self):
        source = textwrap.dedent(
            """
            package com.example;
            public enum Color {
                RED,
                GREEN,
                BLUE;
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "Color.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        self.assertIn("com.example.Color", result.types)
        model = result.types["com.example.Color"]
        self.assertEqual(model.kind, "enum")
        self.assertEqual(model.stereotype, "enumeration")
        self.assertEqual([c.name for c in model.enum_constants], ["RED", "GREEN", "BLUE"])

    def test_parser_extracts_enum_constant_arguments(self):
        source = textwrap.dedent(
            """
            package com.example;
            public enum Status {
                OK(\"All good\"),
                FAIL(\"Nope\");
                private final String message;
                Status(String message) { this.message = message; }
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "Status.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["com.example.Status"]
        self.assertEqual([c.name for c in model.enum_constants], ["OK", "FAIL"])
        self.assertEqual(model.enum_constants[0].arguments, ['"All good"'])
        ctors = [m for m in model.methods if m.is_constructor]
        self.assertTrue(any(c.parameters for c in ctors))

    def test_enum_methods_are_not_duplicated(self):
        source = textwrap.dedent(
            """
            public enum TestEnum {
                TEST1(\"Test1\"),
                TEST2(\"Test2\");

                private String reprezentacia;

                private TestEnum(String reprezentacia) {
                    this.reprezentacia = reprezentacia;
                }

                public String getReprezentacia() {
                    return reprezentacia;
                }
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "TestEnum.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["TestEnum"]
        methods = [m for m in model.methods if (not m.is_constructor and m.name == "getReprezentacia")]
        self.assertEqual(len(methods), 1)

    def test_final_fields_have_stereotype_and_inline_default(self):
        source = textwrap.dedent(
            """
            public class FinalDefaults {
                public final String NAME = \"X\";
                public final int ANSWER = 42;
                public int NON_FINAL = 7;
                public final int assignedInCtor;
                public FinalDefaults() { assignedInCtor = 123; }
            }
            """
        ).strip()

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "FinalDefaults.java")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)

            result = JavaSourceParser().parse_files([path])

        model = result.types["FinalDefaults"]
        fields = {f.name: f for f in model.fields}
        self.assertIn("final", fields["NAME"].modifiers)
        self.assertEqual(fields["NAME"].default_value, '"X"')
        self.assertIn("final", fields["ANSWER"].modifiers)
        self.assertEqual(fields["ANSWER"].default_value, "42")
        self.assertNotIn("final", fields["NON_FINAL"].modifiers)
        self.assertIsNone(fields["NON_FINAL"].default_value)
        self.assertIn("final", fields["assignedInCtor"].modifiers)
        self.assertEqual(fields["assignedInCtor"].default_value, "123")


if __name__ == "__main__":
    unittest.main()
