from zotero_arxiv_daily.utils import glob_match


class TestGlobMatch:
    """Test cases for the glob_match function."""
    
    def test_exact_match(self):
        """Test exact string matching."""
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("hello.txt", "world.txt")
        assert glob_match("", "")
    
    def test_wildcard_asterisk(self):
        """Test asterisk (*) wildcard matching."""
        # Single asterisk
        assert glob_match("hello.txt", "*.txt")
        assert not glob_match("hello.py", "*.txt")
        assert glob_match("file", "*")
        
        # Multiple asterisks
        assert glob_match("hello.world.txt", "*.*.txt")
        assert not glob_match("hello.txt", "*.*.txt")
        assert glob_match("a.b.c.d", "*.*.*.*")
        
        # Asterisk in middle
        assert glob_match("hello_world.txt", "hello*world.txt")
        assert glob_match("hello123world.txt", "hello*world.txt")
        assert glob_match("helloworld.txt", "hello*world.txt")
        assert not glob_match("hello_universe.txt", "hello*world.txt")
    
    def test_wildcard_question_mark(self):
        """Test question mark (?) wildcard matching."""
        assert glob_match("hello.txt", "hell?.txt")
        assert not glob_match("hell.txt", "hell?.txt")  # Missing one character
        assert glob_match("hello.txt", "he??o.txt")
        assert glob_match("heXXo.txt", "he??o.txt")
        assert not glob_match("heo.txt", "he??o.txt")  # Missing characters
    
    def test_character_classes(self):
        """Test character class matching with square brackets."""
        # Basic character class
        assert glob_match("file1.txt", "file[123].txt")
        assert glob_match("file2.txt", "file[123].txt")
        assert not glob_match("file4.txt", "file[123].txt")
        
        # Range character class
        assert glob_match("file1.txt", "file[1-3].txt")
        assert glob_match("file2.txt", "file[1-3].txt")
        assert not glob_match("file4.txt", "file[1-3].txt")
        
        # Negated character class
        assert glob_match("fileA.txt", "file[!123].txt")
        assert not glob_match("file1.txt", "file[!123].txt")
    
    def test_path_separators(self):
        """Test matching with path separators."""
        assert glob_match("dir/file.txt", "dir/file.txt")
        assert glob_match("dir/file.txt", "*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/subdir/file.txt")
        assert glob_match("dir/subdir/file.txt", "dir/*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/*/file.txt")
    
    def test_complex_patterns(self):
        """Test complex glob patterns combining multiple features."""
        # Combining wildcards and character classes
        assert glob_match("test1_file.txt", "test[1-3]*file.txt")
        assert glob_match("test2_long_file.txt", "test[1-3]*file.txt")
        assert not glob_match("test4_file.txt", "test[1-3]*file.txt")
        
        # Multiple wildcards with specific endings
        assert glob_match("prefix_middle_suffix.log", "prefix*middle*.log")
        assert not glob_match("prefix_other_suffix.log", "prefix*middle*.log")
    
    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Empty strings
        assert glob_match("", "**")
        assert not glob_match("", "?")
        assert not glob_match("file", "")
        
        # Special characters in filenames
        assert glob_match("file-name.txt", "file-name.txt")
        assert glob_match("file_name.txt", "file_name.txt")
        assert glob_match("file.name.txt", "file.name.txt")
        
        # Case sensitivity (glob patterns are typically case-sensitive)
        assert not glob_match("File.txt", "file.txt")
        assert not glob_match("FILE.TXT", "file.txt")
        assert glob_match("file.txt", "file.txt")
    
    def test_partial_matches(self):
        """Test that function only matches from the beginning of the string."""
        # The function uses re.match() which only matches from the beginning
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("prefix_hello.txt", "hello.txt")
        assert glob_match("hello.txt", "*.txt")
        
    def test_special_glob_characters(self):
        """Test patterns with special glob characters."""
        # Testing with literal brackets (should be escaped in real usage)
        # Note: This tests the behavior of glob.translate()
        assert not glob_match("file[1].txt", "file[1].txt")  # [1] is treated as character class
        assert glob_match("file1.txt", "file[1].txt")     # [1] matches '1'
        
        # Testing with literal asterisk would require escaping in the pattern
        # This is more about understanding glob.translate() behavior
    
    def test_numeric_patterns(self):
        """Test patterns with numeric components."""
        assert glob_match("file001.txt", "file???.txt")
        assert not glob_match("file01.txt", "file???.txt")
        assert glob_match("version1.2.3.txt", "version*.txt")
        assert glob_match("version1.2.3.txt", "version?.?.?.txt")
        
    def test_extension_patterns(self):
        """Test common file extension patterns."""
        # Common extension matching
        assert glob_match("document.pdf", "*.pdf")
        assert glob_match("image.jpg", "*.jpg")
        assert glob_match("script.py", "*.py")
        assert not glob_match("data.csv", "*.txt")
        
        # Multiple possible extensions
        assert glob_match("file.txt", "*.[tc][xs][tv]")  # matches .txt, .csv, etc.
        assert glob_match("file.csv", "*.[tc][xs][tv]")
        assert not glob_match("file.pdf", "*.[tc][xs][tv]")
    
    def test_recursive_wildcard(self):
        """Test recursive wildcard matching."""
        assert glob_match("file.txt", "**/*.txt")
        assert glob_match("dir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/subsubdir/file.txt", "**/*.txt")
