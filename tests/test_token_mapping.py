# Unit tests for token-to-timestamp mapping
# Place this file at: tests/test_token_mapping.py

import pytest
from src.ingestion.chunker import TokenMapper


class TestTokenToTimestampMapping:
    """Test mapping tokens to timestamps."""
    
    def test_token_timestamp_linear(self):
        """Test simple linear timestamp mapping."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 1.0, "word": "second"},
            {"id": 3, "timestamp": 2.0, "word": "third"},
            {"id": 4, "timestamp": 3.0, "word": "end"},
        ]
        
        # Get timestamp for token 2
        timestamp = mapper.get_timestamp_for_token(tokens, token_id=2)
        assert timestamp == 1.0
    
    def test_token_range_timestamps(self):
        """Test getting timestamp range for token range."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 0.5, "word": "second"},
            {"id": 3, "timestamp": 1.0, "word": "third"},
            {"id": 4, "timestamp": 1.5, "word": "end"},
        ]
        
        start_ts, end_ts = mapper.get_timestamps_for_range(tokens, start_id=2, end_id=4)
        
        assert start_ts == 0.5
        assert end_ts == 1.5
    
    def test_token_not_found(self):
        """Test handling when token ID not found."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "first"},
            {"id": 2, "timestamp": 1.0, "word": "second"},
        ]
        
        # Token 999 doesn't exist
        with pytest.raises(ValueError):
            mapper.get_timestamp_for_token(tokens, token_id=999)
    
    def test_token_interpolation(self):
        """Test timestamp interpolation for missing tokens."""
        mapper = TokenMapper()
        
        # Tokens with gaps in IDs
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 5, "timestamp": 2.0, "word": "end"},  # Gap in IDs
        ]
        
        # Token 3 doesn't exist, but should interpolate
        # Between token 1 (0.0s) and token 5 (2.0s)
        # Token 3 should be at approximately 1.0s
        timestamp = mapper.get_timestamp_for_token(tokens, token_id=3, interpolate=True)
        
        assert timestamp is not None
        assert 0.0 <= timestamp <= 2.0
    
    def test_consecutive_token_mapping(self):
        """Test mapping consecutive token IDs."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 10, "timestamp": 5.0, "word": "word1"},
            {"id": 11, "timestamp": 5.5, "word": "word2"},
            {"id": 12, "timestamp": 6.0, "word": "word3"},
            {"id": 13, "timestamp": 6.5, "word": "word4"},
        ]
        
        for token in tokens:
            ts = mapper.get_timestamp_for_token(tokens, token_id=token["id"])
            assert ts == token["timestamp"]
    
    def test_duration_from_tokens(self):
        """Test calculating duration from token range."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 10.0, "word": "start"},
            {"id": 2, "timestamp": 15.0, "word": "middle"},
            {"id": 3, "timestamp": 20.0, "word": "end"},
        ]
        
        duration = mapper.get_duration_for_range(tokens, start_id=1, end_id=3)
        assert duration == 10.0  # 20.0 - 10.0
    
    def test_reverse_mapping(self):
        """Test mapping timestamp back to token ID."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 1.0, "word": "second"},
            {"id": 3, "timestamp": 2.0, "word": "third"},
            {"id": 4, "timestamp": 3.0, "word": "end"},
        ]
        
        # Find token at timestamp 2.0
        token_id = mapper.get_token_at_timestamp(tokens, timestamp=2.0)
        assert token_id == 3
    
    def test_nearest_token_for_timestamp(self):
        """Test finding nearest token for a given timestamp."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 1.0, "word": "second"},
            {"id": 3, "timestamp": 2.0, "word": "third"},
            {"id": 4, "timestamp": 4.0, "word": "end"},
        ]
        
        # Find nearest token to 2.5s (between token 3 and 4)
        nearest_id = mapper.get_nearest_token_for_timestamp(tokens, timestamp=2.5)
        
        # Should be either 3 or 4
        assert nearest_id in [3, 4]


class TestTimestampEdgeCases:
    """Test edge cases in timestamp mapping."""
    
    def test_zero_timestamp(self):
        """Test handling of zero timestamp."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "first"},
            {"id": 2, "timestamp": 0.5, "word": "second"},
        ]
        
        ts = mapper.get_timestamp_for_token(tokens, token_id=1)
        assert ts == 0.0
    
    def test_identical_timestamps(self):
        """Test handling of identical timestamps."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 1.0, "word": "word1"},
            {"id": 2, "timestamp": 1.0, "word": "word2"},  # Same timestamp
            {"id": 3, "timestamp": 2.0, "word": "word3"},
        ]
        
        # Should handle gracefully
        ts1 = mapper.get_timestamp_for_token(tokens, token_id=1)
        ts2 = mapper.get_timestamp_for_token(tokens, token_id=2)
        
        assert ts1 == 1.0
        assert ts2 == 1.0
    
    def test_non_sequential_tokens(self):
        """Test handling non-sequential token IDs."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 100, "timestamp": 0.0, "word": "first"},
            {"id": 101, "timestamp": 1.0, "word": "second"},
            {"id": 102, "timestamp": 2.0, "word": "third"},
        ]
        
        ts = mapper.get_timestamp_for_token(tokens, token_id=101)
        assert ts == 1.0
    
    def test_unsorted_timestamps_error(self):
        """Test that unsorted timestamps raise error."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 2.0, "word": "out"},
            {"id": 2, "timestamp": 1.0, "word": "of"},
            {"id": 3, "timestamp": 3.0, "word": "order"},
        ]
        
        # Should raise error or warn about unsorted data
        with pytest.raises((ValueError, AssertionError)):
            mapper.validate_token_sequence(tokens)
    
    def test_large_token_ids(self):
        """Test handling of very large token IDs."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1000000, "timestamp": 0.0, "word": "big"},
            {"id": 1000001, "timestamp": 1.0, "word": "ids"},
            {"id": 1000002, "timestamp": 2.0, "word": "work"},
        ]
        
        ts = mapper.get_timestamp_for_token(tokens, token_id=1000001)
        assert ts == 1.0
    
    def test_floating_point_precision(self):
        """Test floating point precision in timestamps."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.123456, "word": "precise"},
            {"id": 2, "timestamp": 1.654321, "word": "timestamps"},
        ]
        
        ts = mapper.get_timestamp_for_token(tokens, token_id=1)
        assert abs(ts - 0.123456) < 0.000001  # Within precision


class TestTokenMappingPerformance:
    """Test performance of token mapping."""
    
    def test_mapping_large_token_list(self):
        """Test performance with large token lists."""
        import time
        
        mapper = TokenMapper()
        
        # Create 100,000 tokens
        tokens = [
            {"id": i, "timestamp": float(i * 0.01), "word": f"word{i}"}
            for i in range(1, 100001)
        ]
        
        start = time.time()
        
        # Do multiple lookups
        for i in [10000, 50000, 99999]:
            mapper.get_timestamp_for_token(tokens, token_id=i)
        
        elapsed = time.time() - start
        
        # Should complete quickly (under 1 second)
        assert elapsed < 1.0
    
    def test_mapping_caching(self):
        """Test that mapping can use caching for performance."""
        mapper = TokenMapper(use_cache=True)
        
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 1001)
        ]
        
        # First lookup (uncached)
        import time
        start = time.time()
        mapper.get_timestamp_for_token(tokens, token_id=500)
        first_time = time.time() - start
        
        # Second lookup (cached)
        start = time.time()
        mapper.get_timestamp_for_token(tokens, token_id=500)
        second_time = time.time() - start
        
        # Cached should be faster (or similar)
        # Second call should be instant
        assert second_time <= first_time * 2


class TestTokenValidation:
    """Test token validation and sanity checks."""
    
    def test_validate_token_structure(self):
        """Test validation of token structure."""
        mapper = TokenMapper()
        
        valid_token = {"id": 1, "timestamp": 0.0, "word": "test"}
        assert mapper.validate_token(valid_token)
        
        # Missing timestamp
        invalid_token = {"id": 1, "word": "test"}
        with pytest.raises(ValueError):
            mapper.validate_token(invalid_token)
    
    def test_validate_token_types(self):
        """Test validation of token data types."""
        mapper = TokenMapper()
        
        # Valid types
        valid_token = {"id": 1, "timestamp": 1.5, "word": "text"}
        assert mapper.validate_token(valid_token)
        
        # Invalid: timestamp as string
        invalid_token = {"id": 1, "timestamp": "1.5", "word": "text"}
        with pytest.raises(TypeError):
            mapper.validate_token(invalid_token)
    
    def test_validate_timestamp_range(self):
        """Test validation of timestamp ranges."""
        mapper = TokenMapper()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 10.0, "word": "end"},
        ]
        
        # Valid range
        assert mapper.validate_timestamp_range(tokens, 0.0, 10.0)
        
        # Invalid range (start > end)
        with pytest.raises(ValueError):
            mapper.validate_timestamp_range(tokens, 10.0, 0.0)
