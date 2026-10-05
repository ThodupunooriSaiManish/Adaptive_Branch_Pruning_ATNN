# Adaptive Branch Pruning for ATNN

## Overview

Continual learning on streaming data requires models to adapt to changing data distributions while retaining previously learned knowledge. The Adaptive Tree-like Neural Network (ATNN) addresses catastrophic forgetting under concept drift by dynamically growing branches for new concepts and adapting to recurring concepts.

However, continuous branch growth can increase model complexity and resource consumption. The base ATNN implementation uses a branch-count limit, but does not explicitly consider the utility, redundancy, and historical or recurrent relevance of individual branches when making branch lifecycle decisions.

This project proposes **Adaptive Branch Lifecycle Management** with utility-based branch retention, reuse, and pruning to reduce unnecessary model growth while preserving classification performance and previously learned knowledge under concept drift.

## Problem Statement

In continual learning environments, data arrives continuously and its underlying distribution may change over time. While ATNN dynamically adapts its architecture to new and recurring concepts, continuous branch growth can increase model complexity, memory usage, and computational cost.

Therefore, this project aims to develop an adaptive branch lifecycle mechanism that evaluates the usefulness, redundancy, usage, and historical relevance of individual branches and determines whether they should be retained, reused, or considered for pruning.

## Objectives

1. Study and reproduce the Adaptive Tree-like Neural Network (ATNN) for continual learning under concept drift.

2. Develop an adaptive branch utility evaluation mechanism based on branch performance, usage, redundancy, and historical or recurrent relevance.

3. Implement and evaluate branch retention, reuse, and pruning strategies to control model growth while preserving classification performance and previously learned knowledge.

## Proposed Approach

Streaming Data
       ↓
Concept Drift Detection
       ↓
New / Recurring Concept Identification
       ↓
ATNN Branch Adaptation
       ↓
Branch Utility Evaluation
       ↓
Retain / Reuse / Prune
       ↓
Updated ATNN Model