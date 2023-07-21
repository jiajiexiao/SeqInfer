import pytest

from seqlearn.utils.misc import import_object_from_path


class TestImportObjectFromPath:
    """Unit test class for `import_object_from_path`"""

    def test_import_builtin_function(self):
        """Test importing a built-in function using import_object_from_path."""
        obj = import_object_from_path("os.path.join")
        assert callable(obj)  # Check if the object is callable (in this case, it's a function).
        assert obj.__name__ == "join"  # Check if the object's name matches the expected name.

    def test_import_module_attribute(self):
        """Test importing an attribute from a module using import_object_from_path."""
        obj = import_object_from_path("math.pi")
        assert (obj - 3.141592653589793) < 1e-16

    def test_import_nonexistent_object(self):
        """Test handling import of a nonexistent object using import_object_from_path."""
        with pytest.raises(
            ImportError, match="Could not import object from path 'nonexistent.module.object':"
        ):
            import_object_from_path("nonexistent.module.object")

    def test_import_nonexistent_module(self):
        """Test handling import from a nonexistent module using import_object_from_path."""
        with pytest.raises(
            ImportError, match="Could not import object from path 'nonexistent_module.object':"
        ):
            import_object_from_path("nonexistent_module.object")

    def test_import_invalid_path_format(self):
        """Test handling import with an invalid path format using import_object_from_path."""
        with pytest.raises(
            ImportError, match="Could not import object from path 'invalid.path.format':"
        ):
            import_object_from_path("invalid.path.format")

    def test_import_attribute_error(self):
        """Test handling import of a non-existent attribute using import_object_from_path."""
        with pytest.raises(
            ImportError,
            match=(
                "Could not import object from path 'os.path.nonexistent_attribute': "
                "module 'posixpath' has no attribute 'nonexistent_attribute'"
            ),
        ):
            import_object_from_path("os.path.nonexistent_attribute")
