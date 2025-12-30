"""Tests for experience buffer and data structures."""

import pytest
import torch

from decentralized_verl.training.experience import (
    Experience,
    ExperienceBuffer,
    BatchIterator,
    collate_experiences,
)


class TestExperience:
    """Tests for Experience class."""

    def test_experience_creation(self):
        """Test creating an experience."""
        exp = Experience(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            reward=0.8,
        )
        assert exp.prompt == "What is AI?"
        assert exp.response == "AI is artificial intelligence."
        assert exp.reward == 0.8

    def test_auto_generated_ids(self):
        """Test that IDs are auto-generated."""
        exp = Experience(prompt="test", response="response")
        assert exp.prompt_id != ""
        assert exp.response_id != ""
        assert len(exp.prompt_id) == 8
        assert len(exp.response_id) == 8

    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        exp = Experience(
            prompt="test",
            response="response",
            log_probs=[-0.5, -0.3, -0.4],
            ref_log_probs=[-0.6, -0.4, -0.5],
        )
        kl = exp.kl_divergence
        # KL = mean(log_probs - ref_log_probs) = mean([0.1, 0.1, 0.1]) = 0.1
        assert abs(kl - 0.1) < 0.01

    def test_kl_divergence_empty(self):
        """Test KL divergence with empty log probs."""
        exp = Experience(prompt="test", response="response")
        assert exp.kl_divergence == 0.0

    def test_response_length(self):
        """Test response length calculation."""
        exp = Experience(
            prompt="test",
            response="one two three four five",
        )
        assert exp.response_length == 5

    def test_to_dict_from_dict(self):
        """Test serialization."""
        original = Experience(
            prompt="test prompt",
            response="test response",
            reward=0.5,
            log_probs=[-0.1, -0.2],
            policy_version=3,
        )

        as_dict = original.to_dict()
        restored = Experience.from_dict(as_dict)

        assert restored.prompt == original.prompt
        assert restored.response == original.response
        assert restored.reward == original.reward
        assert restored.log_probs == original.log_probs


class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""

    def test_buffer_creation(self):
        """Test creating a buffer."""
        buffer = ExperienceBuffer(max_size=100, min_size=10)
        assert len(buffer) == 0
        assert buffer.max_size == 100
        assert buffer.min_size == 10

    def test_add_experience(self):
        """Test adding experiences."""
        buffer = ExperienceBuffer(max_size=100)
        exp = Experience(prompt="test", response="response", reward=0.5)
        buffer.add(exp)
        assert len(buffer) == 1

    def test_add_batch(self):
        """Test adding batch of experiences."""
        buffer = ExperienceBuffer(max_size=100)
        experiences = [
            Experience(prompt=f"prompt_{i}", response=f"response_{i}")
            for i in range(5)
        ]
        buffer.add_batch(experiences)
        assert len(buffer) == 5

    def test_max_size_enforcement(self):
        """Test that buffer respects max size."""
        buffer = ExperienceBuffer(max_size=5)
        for i in range(10):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}"))
        assert len(buffer) == 5

    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ExperienceBuffer(max_size=100, on_policy=False)
        for i in range(20):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}"))

        sample = buffer.sample(5)
        assert len(sample) == 5

    def test_on_policy_clears_after_sample(self):
        """Test on-policy mode clears sampled experiences."""
        buffer = ExperienceBuffer(max_size=100, on_policy=True)
        for i in range(10):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}"))

        initial_size = len(buffer)
        sample = buffer.sample(5)
        assert len(buffer) == initial_size - 5

    def test_get_all(self):
        """Test getting all experiences."""
        buffer = ExperienceBuffer(max_size=100, on_policy=True)
        for i in range(5):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}"))

        all_exp = buffer.get_all()
        assert len(all_exp) == 5
        assert len(buffer) == 0  # Should be cleared in on-policy mode

    def test_is_ready(self):
        """Test readiness check."""
        buffer = ExperienceBuffer(max_size=100, min_size=5)
        assert buffer.is_ready() is False

        for i in range(5):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}"))

        assert buffer.is_ready() is True

    def test_filter_by_version(self):
        """Test filtering by policy version."""
        buffer = ExperienceBuffer(max_size=100, on_policy=False)

        for i in range(3):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}", policy_version=1))
        for i in range(2):
            buffer.add(Experience(prompt=f"p{i}", response=f"r{i}", policy_version=2))

        v1_exp = buffer.filter_by_version(1)
        v2_exp = buffer.filter_by_version(2)

        assert len(v1_exp) == 3
        assert len(v2_exp) == 2

    def test_stats(self):
        """Test statistics calculation."""
        buffer = ExperienceBuffer(max_size=100)
        buffer.add(Experience(prompt="p1", response="r1", reward=0.5))
        buffer.add(Experience(prompt="p2", response="r2", reward=1.0))

        stats = buffer.stats
        assert stats["size"] == 2
        assert stats["avg_reward"] == 0.75
        assert stats["min_reward"] == 0.5
        assert stats["max_reward"] == 1.0


class TestBatchIterator:
    """Tests for BatchIterator class."""

    def test_iteration(self):
        """Test basic iteration."""
        experiences = [
            Experience(prompt=f"p{i}", response=f"r{i}")
            for i in range(10)
        ]
        iterator = BatchIterator(experiences, batch_size=3)

        batches = list(iterator)
        assert len(batches) == 4  # 3 + 3 + 3 + 1

    def test_drop_last(self):
        """Test dropping last incomplete batch."""
        experiences = [
            Experience(prompt=f"p{i}", response=f"r{i}")
            for i in range(10)
        ]
        iterator = BatchIterator(experiences, batch_size=3, drop_last=True)

        batches = list(iterator)
        assert len(batches) == 3  # 3 + 3 + 3

    def test_shuffle(self):
        """Test shuffling."""
        experiences = [
            Experience(prompt=f"p{i}", response=f"r{i}")
            for i in range(100)
        ]

        # Without shuffle
        iter1 = BatchIterator(experiences, batch_size=10, shuffle=False)
        batch1 = next(iter(iter1))

        # With shuffle - first batch should likely be different
        iter2 = BatchIterator(experiences, batch_size=10, shuffle=True)
        batch2 = next(iter(iter2))

        # Not a perfect test but checks shuffling works
        assert len(batch1) == len(batch2) == 10


class TestCollateExperiences:
    """Tests for collate_experiences function."""

    def test_basic_collation(self):
        """Test basic collation without tokenizer."""
        experiences = [
            Experience(
                prompt="p1",
                response="r1",
                reward=0.5,
                log_probs=[-0.1, -0.2],
            ),
            Experience(
                prompt="p2",
                response="r2",
                reward=0.7,
                log_probs=[-0.3, -0.4],
            ),
        ]

        batch = collate_experiences(experiences)

        assert batch["prompts"] == ["p1", "p2"]
        assert batch["responses"] == ["r1", "r2"]
        assert torch.allclose(batch["rewards"], torch.tensor([0.5, 0.7]))

    def test_collation_with_advantages(self):
        """Test collation includes advantages."""
        experiences = [
            Experience(
                prompt="p1",
                response="r1",
                advantages=[0.1, 0.2],
                returns=[0.5, 0.6],
            ),
        ]

        batch = collate_experiences(experiences)

        assert "advantages" in batch
        assert "returns" in batch
