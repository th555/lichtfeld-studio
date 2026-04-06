/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <SDL3/SDL.h>

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "visualizer/core/data_loading_service.hpp"
#include "visualizer/visualizer_impl.hpp"
#include "visualizer/include/visualizer/visualizer.hpp"

#include <filesystem>
#include <gtest/gtest.h>

TEST(VisualizerPostWorkTest, QueuedWorkWakesEventLoop) {
    ASSERT_TRUE(SDL_Init(SDL_INIT_EVENTS));
    SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;

    bool ran = false;
    {
        auto viewer = lfs::vis::Visualizer::create(options);

        EXPECT_FALSE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
        EXPECT_TRUE(viewer->postWork({
            .run = [&ran]() { ran = true; },
            .cancel = nullptr,
        }));

        EXPECT_FALSE(ran);
        EXPECT_TRUE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
    }
}

class VisualizerImplResetTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
    }

    void TearDown() override {
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
    }
};

namespace lfs::vis {

TEST_F(VisualizerImplResetTest, ResetTrainingPreservesExplicitInitPath) {
    ViewerOptions options;
    options.show_startup_overlay = false;

    const auto dataset_path = std::filesystem::temp_directory_path() / "lfs_reset_preserves_init_dataset";
    std::filesystem::create_directories(dataset_path);

    VisualizerImpl viewer(options);
    viewer.getSceneManager()->changeContentType(SceneManager::ContentType::Dataset);
    viewer.getSceneManager()->setDatasetPath(dataset_path);

    lfs::core::param::TrainingParameters params;
    params.init_path = "seed_points.ply";
    viewer.getDataLoader()->setParameters(params);

    viewer.performReset();

    ASSERT_TRUE(viewer.getDataLoader()->getParameters().init_path.has_value());
    EXPECT_EQ(*viewer.getDataLoader()->getParameters().init_path, "seed_points.ply");

    std::error_code ec;
    std::filesystem::remove_all(dataset_path, ec);
}

} // namespace lfs::vis
