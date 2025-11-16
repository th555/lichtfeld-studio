/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dataset.hpp"
#include "trainer.hpp"
#include "core_new/tensor.hpp"
#include <expected>
#include <memory>

namespace lfs::training {
    struct TrainingSetup {
        std::unique_ptr<Trainer> trainer;
        std::shared_ptr<CameraDataset> dataset;
        lfs::core::Tensor scene_center;
    };

    // Reusable function to set up training from parameters
    std::expected<TrainingSetup, std::string> setupTraining(const lfs::core::param::TrainingParameters& params);
} // namespace lfs::training
