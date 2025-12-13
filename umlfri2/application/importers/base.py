from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from umlfri2.application import Application
from umlfri2.model import ElementObject, Project
from umlfri2.model.diagram import Diagram
from umlfri2.types.geometry import Point, Size

LOGGER = logging.getLogger(__name__)

INFJAVA_UML_ADDON_ID = "urn:umlfri.org:metamodel:infjavauml"
DEFAULT_TEMPLATE_ID = "empty"
DEFAULT_DIAGRAM_TYPE = "class_diagram"
DEFAULT_CLASS_ELEMENT = "class"
DEFAULT_PACKAGE_ELEMENT = "package"
GENERALISATION_CONNECTION = "generalisation"
IMPLEMENTATION_CONNECTION = "implementation"
ASSOCIATION_CONNECTION = "association"
COMPOSITION_CONNECTION = "composition"

VISIBILITY_MAP = {
    "public": "+",
    "private": "-",
    "protected": "#",
    "internal": "~",
}


class ImportError(Exception):
    """Raised when the importer cannot complete the requested action."""


class ImportView(Enum):
    """Specifies which members to include when importing sources."""
    INTERNAL = "internal"  # All members (private, protected, public)
    EXTERNAL = "external"  # Only public members


@dataclass
class TypeDescriptor:
    name: str
    qualifier: Optional[str] = None
    arguments: List["TypeDescriptor"] = field(default_factory=list)
    dimensions: int = 0

    def display(self) -> str:
        base = "".join(filter(None, [self.qualifier, "." if self.qualifier else "", self.name]))
        if not base:
            base = self.name
        if self.arguments:
            # Filter out None arguments
            valid_args = [arg for arg in self.arguments if arg is not None]
            if valid_args:
                base = base + "<" + ", ".join(arg.display() for arg in valid_args) + ">"
        if self.dimensions:
            base = base + "[]" * self.dimensions
        return base

    def direct_full_name(self) -> Optional[str]:
        if self.qualifier:
            return f"{self.qualifier}.{self.name}"
        return None


@dataclass
class FieldModel:
    name: str
    type_descriptor: Optional[TypeDescriptor]
    modifiers: Set[str]
    is_instantiated: bool = False
    is_assigned_externally: bool = False

    @property
    def visibility(self) -> str:
        for mod in self.modifiers:
            if mod in VISIBILITY_MAP:
                return VISIBILITY_MAP[mod]
        return "-"

    @property
    def is_static(self) -> bool:
        return "static" in self.modifiers


@dataclass
class MethodParameter:
    name: str
    type_descriptor: Optional[TypeDescriptor]


@dataclass
class MethodModel:
    name: str
    return_type: Optional[TypeDescriptor]
    parameters: List[MethodParameter]
    modifiers: Set[str]
    is_constructor: bool = False

    @property
    def visibility(self) -> str:
        for mod in self.modifiers:
            if mod in VISIBILITY_MAP:
                return VISIBILITY_MAP[mod]
        return "-"

    @property
    def is_static(self) -> bool:
        return "static" in self.modifiers

    @property
    def is_abstract(self) -> bool:
        return "abstract" in self.modifiers


@dataclass
class TypeModel:
    name: str
    package: Optional[str]
    kind: str  # "class", "interface", "enum", "struct"
    modifiers: Set[str]
    fields: List[FieldModel]
    methods: List[MethodModel]
    extends: List[TypeDescriptor]
    implements: List[TypeDescriptor]
    source_path: str
    enum_constants: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        if self.package:
            return f"{self.package}.{self.name}"
        return self.name

    @property
    def is_abstract(self) -> bool:
        return "abstract" in self.modifiers

    @property
    def stereotype(self) -> Optional[str]:
        if self.kind == "interface":
            return "interface"
        if self.kind == "enum":
            return "enumeration"
        if self.kind == "struct":
            return "struct"
        return None


@dataclass
class ParseResult:
    types: Dict[str, TypeModel]
    errors: List[str]


@dataclass
class BuildSummary:
    elements_created: int
    connections_created: int
    primary_diagram: Optional[Diagram]


@dataclass
class ImportReport:
    summary: BuildSummary
    warnings: List[str]


class TypeResolver:
    def __init__(self, types: Dict[str, TypeModel]):
        self._types = types
        self._simple_index: Dict[str, Set[str]] = defaultdict(set)
        for full_name, model in types.items():
            self._simple_index[model.name].add(full_name)

    def resolve(self, descriptor: Optional[TypeDescriptor], context: TypeModel) -> Optional[str]:
        if descriptor is None:
            return None
        direct = descriptor.direct_full_name()
        if direct and direct in self._types:
            return direct
        simple = descriptor.name
        if context.package:
            candidate = f"{context.package}.{simple}"
            if candidate in self._types:
                return candidate
        matches = self._simple_index.get(simple)
        if matches and len(matches) == 1:
            return next(iter(matches))
        if direct:
            return direct
        return None


class BaseModelBuilder:
    BASE_ELEMENT_WIDTH = 240
    BASE_ELEMENT_HEIGHT = 140
    ELEMENT_SPACING_X = 80
    ELEMENT_SPACING_Y = 100
    LAYER_MARGIN = 60
    MIN_NODE_SEPARATION = 40

    def __init__(self, project: Project, ruler, view: ImportView = ImportView.INTERNAL):
        self._project = project
        self._ruler = ruler
        self._view = view
        self._metamodel = project.metamodel
        self._package_type = self._metamodel.get_element_type(DEFAULT_PACKAGE_ELEMENT)
        self._class_type = self._metamodel.get_element_type(DEFAULT_CLASS_ELEMENT)
        self._diagram_type = self._metamodel.get_diagram_type(DEFAULT_DIAGRAM_TYPE)
        self._generalisation_type = self._metamodel.get_connection_type(GENERALISATION_CONNECTION)
        self._implementation_type = self._metamodel.get_connection_type(IMPLEMENTATION_CONNECTION)
        self._association_type = self._metamodel.get_connection_type(ASSOCIATION_CONNECTION)
        self._composition_type = self._metamodel.get_connection_type(COMPOSITION_CONNECTION)
        self._packages: Dict[Tuple[str, ...], ElementObject] = {}
        self._class_elements: Dict[str, ElementObject] = {}
        self._class_visuals: Dict[str, object] = {}
        self._connections: Set[Tuple[str, str, str]] = set()
        self._diagram: Optional[Diagram] = None

    def build(self, types: Dict[str, TypeModel]) -> BuildSummary:
        if not types:
            raise ImportError("No types to import")
        resolver = TypeResolver(types)
        root_package = self._create_root_package()
        diagram = self._ensure_root_diagram(root_package)
        self._diagram = diagram

        package_groups: Dict[Tuple[str, ...], List[TypeModel]] = defaultdict(list)
        for model in types.values():
            path = tuple(model.package.split(".")) if model.package else tuple()
            package_groups[path].append(model)

        ordered_packages = sorted(package_groups.items(), key=lambda item: (len(item[0]), ".".join(item[0])))
        total_elements = 0
        for package_path, models in ordered_packages:
            package_element = self._ensure_package(package_path, root_package)
            for model in models:
                element = self._create_class_element(package_element, model)
                self._class_elements[model.full_name] = element
                self._register_class_visual(model.full_name, element)
                total_elements += 1

        total_connections = self._create_connections(types, resolver, diagram)
        self._layout_classes(types, resolver)
        self._project.invalidate_all_caches()
        return BuildSummary(elements_created=total_elements, connections_created=total_connections, primary_diagram=diagram)

    def _create_root_package(self):
        package = self._project.create_child_element(self._package_type)
        mutable = package.data.make_mutable()
        mutable.set_value("name", self._project.name or "Imported Model")
        package.apply_ufl_patch(mutable.make_patch())
        self._packages[tuple()] = package
        return package

    def _ensure_package(self, path: Tuple[str, ...], root_package):
        if not path:
            return self._packages[tuple()]
        current = root_package
        built_path: List[str] = []
        for segment in path:
            built_path.append(segment)
            tuple_path = tuple(built_path)
            if tuple_path not in self._packages:
                pkg = current.create_child_element(self._package_type)
                mutable = pkg.data.make_mutable()
                mutable.set_value("name", segment)
                pkg.apply_ufl_patch(mutable.make_patch())
                self._packages[tuple_path] = pkg
            current = self._packages[tuple_path]
        return current

    def _ensure_root_diagram(self, root_package):
        diagram = root_package.create_child_diagram(self._diagram_type)
        mutable = diagram.data.make_mutable()
        mutable.set_value("name", "Imported Classes")
        diagram.apply_ufl_patch(mutable.make_patch())
        return diagram

    def _create_class_element(self, parent, model: TypeModel):
        element = parent.create_child_element(self._class_type)
        mutable = element.data.make_mutable()
        mutable.set_value("name", model.name)
        mutable.set_value("abstract", model.is_abstract)
        if model.stereotype:
            mutable.set_value("stereotype", model.stereotype)
        if model.kind == "enum" and model.enum_constants:
            attributes = mutable.get_value("attributes")
            for const in model.enum_constants:
                row = attributes.append()
                row.set_value("name", const)
        else:
            self._populate_attributes(mutable, model.fields)
        self._populate_operations(mutable, model.methods, model.name, is_interface=(model.kind == "interface"))
        element.apply_ufl_patch(mutable.make_patch())
        return element

    def _register_class_visual(self, class_name: str, element):
        if self._diagram is None:
            raise ImportError("Diagram is not initialized")
        visual = self._diagram.show(element)
        self._class_visuals[class_name] = visual

    def _populate_attributes(self, mutable, fields: List[FieldModel]):
        attributes = mutable.get_value("attributes")
        for field in fields:
            if self._view == ImportView.EXTERNAL and field.visibility != "+":
                continue
            row = attributes.append()
            row.set_value("name", field.name)
            if field.type_descriptor:
                row.set_value("type", field.type_descriptor.display())
            row.set_value("visibility", field.visibility)
            if field.is_static:
                row.set_value("static", True)

    def _populate_operations(self, mutable, methods: List[MethodModel], class_name: str = "", is_interface: bool = False):
        operations = mutable.get_value("operations")
        for method in methods:
            if self._view == ImportView.EXTERNAL and method.visibility != "+":
                continue
            row = operations.append()
            if method.is_constructor and self._view == ImportView.EXTERNAL:
                row.set_value("name", "new")
                row.set_value("rtype", class_name)
                row.set_value("static", True)
            else:
                row.set_value("name", method.name)
                if method.return_type:
                    row.set_value("rtype", method.return_type.display())
            row.set_value("visibility", method.visibility)
            if method.is_static:
                row.set_value("static", True)
            if method.is_abstract or is_interface:
                row.set_value("abstract", True)
            
            # Add parameters
            params = row.get_value("parameters")
            for param in method.parameters:
                prow = params.append()
                prow.set_value("name", param.name)
                if param.type_descriptor:
                    prow.set_value("type", param.type_descriptor.display())

    def _layout_classes(self, types: Dict[str, TypeModel], resolver: TypeResolver):
        if not self._class_visuals:
            return
        # Simple grid layout
        x, y = self.LAYER_MARGIN, self.LAYER_MARGIN
        max_row_height = 0
        row_count = 0
        max_per_row = max(3, int(len(types) ** 0.5))
        
        for name in sorted(types.keys()):
            if name not in self._class_visuals:
                continue
            visual = self._class_visuals[name]
            visual.move(self._ruler, Point(x, y))
            size = visual.get_size(self._ruler)
            max_row_height = max(max_row_height, size.height)
            row_count += 1
            if row_count >= max_per_row:
                x = self.LAYER_MARGIN
                y += max_row_height + self.ELEMENT_SPACING_Y
                max_row_height = 0
                row_count = 0
            else:
                x += self.BASE_ELEMENT_WIDTH + self.ELEMENT_SPACING_X

    def _create_connections(self, types: Dict[str, TypeModel], resolver: TypeResolver, diagram) -> int:
        created = 0
        for model in types.values():
            source = self._class_elements.get(model.full_name)
            if source is None:
                continue
            # Create generalization connections
            for ext in model.extends:
                target_name = resolver.resolve(ext, model)
                if target_name and target_name in self._class_elements:
                    target = self._class_elements[target_name]
                    created += self._ensure_connection(source, target, self._generalisation_type, diagram)
            # Create implementation connections
            for impl in model.implements:
                target_name = resolver.resolve(impl, model)
                if target_name and target_name in self._class_elements:
                    target = self._class_elements[target_name]
                    # For interfaces implementing interfaces, use generalization
                    connection_type = self._implementation_type if model.kind == "class" else self._generalisation_type
                    created += self._ensure_connection(source, target, connection_type, diagram)
            # Create associations based on field types
            created += self._create_associations(model, resolver, source, diagram)
        return created

    def _create_associations(self, model: TypeModel, resolver: TypeResolver, source, diagram) -> int:
        """Create association/composition connections based on field types."""
        created = 0
        for field_model in model.fields:
            descriptor = field_model.type_descriptor
            if descriptor is None:
                continue
            
            # Resolve the field type to find potential targets
            association_targets = self._resolve_field_targets(descriptor, resolver, model)
            
            # Determine connection type: composition if field is instantiated inline
            # and not assigned externally (i.e., the class owns the object's lifecycle)
            if field_model.is_instantiated and not field_model.is_assigned_externally and not field_model.is_static:
                connection_type = self._composition_type
                # For composition, the containing class (source) is the "whole"
                # and the field type (target) is the "part"
                for target_name in association_targets:
                    if target_name in self._class_elements:
                        target = self._class_elements[target_name]
                        # Composition arrow points from part to whole
                        created += self._ensure_connection(target, source, connection_type, diagram)
            else:
                connection_type = self._association_type
                for target_name in association_targets:
                    if target_name in self._class_elements:
                        target = self._class_elements[target_name]
                        created += self._ensure_connection(source, target, connection_type, diagram)
        return created

    def _resolve_field_targets(self, descriptor: TypeDescriptor, resolver: TypeResolver, context: TypeModel) -> Set[str]:
        """Resolve a type descriptor to potential association targets."""
        targets: Set[str] = set()
        
        # Try to resolve the direct type
        direct = resolver.resolve(descriptor, context)
        if direct:
            targets.add(direct)
        
        # Also check generic type arguments (e.g., List<Person> -> Person)
        for argument in descriptor.arguments:
            resolved = resolver.resolve(argument, context)
            if resolved:
                targets.add(resolved)
        
        return targets

    def _ensure_connection(self, source, target, connection_type, diagram) -> int:
        key = (connection_type.id, str(source.save_id), str(target.save_id))
        if key in self._connections:
            return 0
        connection = source.connect_with(connection_type, target)
        mutable = connection.data.make_mutable()
        connection.apply_ufl_patch(mutable.make_patch())
        diagram.show(connection)
        self._connections.add(key)
        return 1


class BaseImportController:
    """Base class for source code importers."""

    def __init__(self, application: Optional[Application] = None, addon_identifier: str = INFJAVA_UML_ADDON_ID,
                 template_id: str = DEFAULT_TEMPLATE_ID):
        self._application = application or Application()
        self._addon_identifier = addon_identifier
        self._template_id = template_id

    def _collect_files(self, path: str, extension: str) -> List[str]:
        normalized = os.path.abspath(path)
        if os.path.isfile(normalized) and normalized.endswith(extension):
            return [normalized]
        files: List[str] = []
        for root, _, filenames in os.walk(normalized):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return sorted(files)

    def _find_template(self, templates) -> Optional[object]:
        for template in templates:
            if template.id == self._template_id:
                return template
        return None

    def _create_project(self, name: str) -> Project:
        addon = self._application.addons.local.get_addon(self._addon_identifier)
        if addon is None or addon.metamodel is None:
            raise ImportError(f"Required add-on '{self._addon_identifier}' not found or not loaded")
        template = self._find_template(addon.metamodel.templates)
        if template is None:
            raise ImportError(f"Template '{self._template_id}' not found")
        self._application.new_project(template, new_solution=True, project_name=name)
        return next(self._application.solution.children)
