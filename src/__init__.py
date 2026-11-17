# Chameleon KV-Cache Eviction Benchmark — Package Exports
from src.core import (
    Policy, RandomPolicy, SlidingWindow, StreamingLLM,
    HeavyHitter, Hybrid, BeladyGreedy, ScheduledPolicy, FullCache,
    optimal_schedule, get_optimum, simulate, teacher_forced, synthetic_attention,
)
from src.chameleon import Chameleon, LastAttention, EXPERTS
from src.hybrids import HitLRU, AdaptiveWindow, HARP
from src.imitation import ImitationScorer, train_imitation, dagger_train, state_auc
from src.gauntlet import registry, selftest, stress_traces, adversary
