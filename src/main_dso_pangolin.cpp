/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include <atomic>
#include <cstddef>
#include <locale.h>
#include <memory>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <vector>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "util/NumType.h"

#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"

#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

namespace {

class Config {
public:
  static void SetParameterFile(const std::string &filename) {
    if (config_ == nullptr)
      config_ = std::shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->file_.isOpened() == false) {
      std::cerr << "Parameter file " << filename << " does not exist."
                << std::endl;
      config_->file_.release();
      std::terminate();
    }
  }
  template <typename T>
  static std::optional<T> MaybeGet(const std::string &key) {
    const auto &file_node = Config::config_->file_[key];
    if (file_node.empty()) {
      std::cerr << "Can't find key: " << key << "\n";
      return std::nullopt;
    }
    return std::optional<T>(static_cast<T>(file_node));
  }
  ~Config() {
    if (file_.isOpened())
      file_.release();
  }

private:
  static std::shared_ptr<Config> config_;
  cv::FileStorage file_;
};

std::shared_ptr<Config> Config::config_ = nullptr;

static const char *kVignettePathKey = "vignette_path";
static const char *kImagesPathKey = "images_path";
static const char *kCameraCalibPathKey = "camera_calib_path";
static const char *kGammaCalibPathKey = "gamma_calib_path";
static const char *kReversePlayKey = "reverse_play";
static const char *kRunQuiet = "run_quiet";
static const char *kStartFrameIdx = "start_frame_idx";
static const char *kEndFrameIdx = "end_frame_idx";
static const char *kLoadAllImagesOnce = "load_all_images_once";
static const char *kLogging = "logging";
static const char *kDisableDisplay = "disable_display";
static const char *kMultiThreading = "multi_threading";
static const char *kSaveDebugImage = "save_debug_images";

struct Options {
  std::string images_path = "";
  std::string vignette_path = "";
  std::string camera_calib_path = "";
  std::string gamma_calib_path = "";

  bool reverse_play = false;
  // Start from 0, inclusive.
  int start_frame_idx = 0;
  // Exclusive.
  int end_frame_idx = 10000;
  bool load_all_images_once = false;

  bool setting_debugout_runquiet = true;
  bool setting_log_stuff = false;
  bool setting_disable_display = false;
  bool setting_multi_threading = true;
};

Options ParseConfig(const std::string &config_path) {
  Options options;
  Config::SetParameterFile(config_path);
  const auto maybe_images_path = Config::MaybeGet<std::string>(kImagesPathKey);
  if (maybe_images_path.has_value()) {
    options.images_path = maybe_images_path.value();
  }
  const auto maybe_vignette_path =
      Config::MaybeGet<std::string>(kVignettePathKey);
  if (maybe_vignette_path.has_value()) {
    options.vignette_path = maybe_vignette_path.value();
  }
  const auto maybe_camera_calib_path =
      Config::MaybeGet<std::string>(kCameraCalibPathKey);
  if (maybe_camera_calib_path.has_value()) {
    options.camera_calib_path = maybe_camera_calib_path.value();
  }
  const auto maybe_gamma_calib_path =
      Config::MaybeGet<std::string>(kGammaCalibPathKey);
  if (maybe_gamma_calib_path.has_value()) {
    options.gamma_calib_path = maybe_gamma_calib_path.value();
  }
  const auto maybe_reverse_play = Config::MaybeGet<int>(kReversePlayKey);
  if (maybe_reverse_play.has_value()) {
    options.reverse_play = static_cast<bool>(maybe_reverse_play.value());
  }
  const auto maybe_run_quiet = Config::MaybeGet<int>(kRunQuiet);
  if (maybe_run_quiet.has_value()) {
    options.setting_debugout_runquiet =
        static_cast<bool>(maybe_run_quiet.value());
  }
  const auto maybe_start_frame_idx = Config::MaybeGet<int>(kStartFrameIdx);
  if (maybe_start_frame_idx.has_value()) {
    options.start_frame_idx = std::max(0, maybe_start_frame_idx.value());
  }
  const auto maybe_end_frame_idx = Config::MaybeGet<int>(kEndFrameIdx);
  if (maybe_end_frame_idx.has_value()) {
    options.end_frame_idx = maybe_end_frame_idx.value();
  }
  const auto maybe_load_all_images_once =
      Config::MaybeGet<int>(kLoadAllImagesOnce);
  if (maybe_load_all_images_once.has_value()) {
    options.load_all_images_once =
        static_cast<bool>(maybe_load_all_images_once.value());
  }
  const auto maybe_setting_log_stuff = Config::MaybeGet<int>(kLogging);
  if (maybe_setting_log_stuff.has_value()) {
    options.setting_log_stuff =
        static_cast<bool>(maybe_setting_log_stuff.value());
  }
  const auto maybe_disable_display_stuff =
      Config::MaybeGet<int>(kDisableDisplay);
  if (maybe_disable_display_stuff.has_value()) {
    options.setting_disable_display =
        static_cast<bool>(maybe_disable_display_stuff.value());
  }
  const auto maybe_multi_threading = Config::MaybeGet<int>(kMultiThreading);
  if (maybe_multi_threading.has_value()) {
    options.setting_multi_threading =
        static_cast<bool>(maybe_multi_threading.value());
  }
  return options;
}

void ConfigDsoSettings(const Options &options) {
  // See in util/settings.h.
  setting_debugout_runquiet = options.setting_debugout_runquiet;
  setting_logStuff = options.setting_log_stuff;
  disableAllDisplay = options.setting_disable_display;
  multiThreading = options.setting_multi_threading;
}

} // namespace

// 0 for linearize (play as fast as possible, while sequentializing
// tracking & mapping). otherwise, factor on timestamps.
float playbackSpeed = 0;
bool useSampleOutput = false;

std::atomic<bool> exThreadKeepRunning(true);

int mode = 0;

void my_exit_handler(int s) {
  printf("Caught signal %d\n", s);
  exit(1);
}

void exitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  while (exThreadKeepRunning) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void settingsDefault(int preset) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 0 || preset == 1) {
    printf("DEFAULT settings:\n"
           "- %s real-time enforcing\n"
           "- 2000 active points\n"
           "- 5-7 active frames\n"
           "- 1-6 LM iteration each KF\n"
           "- original image resolution\n",
           preset == 0 ? "no " : "1x");

    playbackSpeed = (preset == 0 ? 0 : 1);
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    setting_logStuff = false;
  }

  if (preset == 2 || preset == 3) {
    printf("FAST settings:\n"
           "- %s real-time enforcing\n"
           "- 800 active points\n"
           "- 4-6 active frames\n"
           "- 1-4 LM iteration each KF\n"
           "- 424 x 320 image resolution\n",
           preset == 0 ? "no " : "5x");

    playbackSpeed = (preset == 2 ? 0 : 5);
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;

    setting_logStuff = false;
  }

  printf("==============================================\n");
}

void parseArgument(char *arg) {
  int option;
  float foption;
  char buf[1000];

  if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
    if (option == 1) {
      useSampleOutput = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "preset=%d", &option)) {
    settingsDefault(option);
    return;
  }

  if (1 == sscanf(arg, "speed=%f", &foption)) {
    playbackSpeed = foption;
    printf("PLAYBACK SPEED %f!\n", playbackSpeed);
    return;
  }

  if (1 == sscanf(arg, "save=%d", &option)) {
    if (option == 1) {
      debugSaveImages = true;
      if (42 == system("rm -rf images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("rm -rf images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      if (42 == system("mkdir images_out"))
        printf("system call returned 42 - what are the odds?. This is only "
               "here to shut up the compiler.\n");
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option)) {

    mode = option;
    if (option == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA =
          -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB =
          -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}

namespace dso {
int Main(int argc, char **argv) {
  for (int i = 1; i < argc; i++)
    parseArgument(argv[i]);
  const auto options = ParseConfig(argv[1]);
  ConfigDsoSettings(options);

  // Hook crtl+C.
  std::thread exThread(exitThread);

  std::unique_ptr<ImageFolderReader> reader(
      new ImageFolderReader(options.images_path, options.camera_calib_path,
                            options.gamma_calib_path, options.vignette_path));

  reader->setGlobalCalibration();

  if (setting_photometricCalibration > 0 &&
      reader->getPhotometricGamma() == 0) {
    std::cerr << "ERROR: don't have photometric calibration. "
              << "Need to use command line options mode = 1 or mode = 2 \n ";
    exit(1);
  }

  int lstart = options.start_frame_idx;
  int lend = options.end_frame_idx;
  int linc = 1;
  if (options.reverse_play) {
    lstart = options.end_frame_idx - 1;
    if (lstart >= reader->getNumImages())
      lstart = reader->getNumImages() - 1;
    lend = options.start_frame_idx;
    linc = -1;
  }

  std::unique_ptr<FullSystem> fullSystem(new FullSystem());
  fullSystem->setGammaFunction(reader->getPhotometricGamma());
  fullSystem->linearizeOperation = (playbackSpeed == 0);

  IOWrap::PangolinDSOViewer *viewer = nullptr;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
    fullSystem->outputWrapper.push_back(viewer);
  }

  if (useSampleOutput) {
    fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());
  }

  // to make MacOS happy: run this in dedicated thread -- and use this one to
  // run the GUI.
  std::thread runthread([&]() {
    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt;
    for (int i = lstart;
         i >= 0 && i < reader->getNumImages() && linc * i < linc * lend;
         i += linc) {
      idsToPlay.push_back(i);
      if (timesToPlayAt.size() == 0) {
        timesToPlayAt.push_back((double)0);
      } else {
        double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
        double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
        timesToPlayAt.push_back(timesToPlayAt.back() +
                                fabs(tsThis - tsPrev) / playbackSpeed);
      }
    }

    std::vector<ImageAndExposure *> preloadedImages;
    if (options.load_all_images_once) {
      printf("LOADING ALL IMAGES!\n");
      for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
        int i = idsToPlay[ii];
        preloadedImages.push_back(reader->getImage(i));
      }
    }

    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();
    double sInitializerOffset = 0;

    for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
      if (!fullSystem->initialized) // if not initialized: reset start time.
      {
        gettimeofday(&tv_start, NULL);
        started = clock();
        sInitializerOffset = timesToPlayAt[ii];
      }

      int i = idsToPlay[ii];

      ImageAndExposure *img;
      if (options.load_all_images_once)
        img = preloadedImages[ii];
      else
        img = reader->getImage(i);

      bool skipFrame = false;
      if (playbackSpeed != 0) {
        struct timeval tv_now;
        gettimeofday(&tv_now, NULL);
        double sSinceStart =
            sInitializerOffset +
            ((tv_now.tv_sec - tv_start.tv_sec) +
             (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

        if (sSinceStart < timesToPlayAt[ii])
          usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
        else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
          printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii,
                 timesToPlayAt[ii], sSinceStart);
          skipFrame = true;
        }
      }
      if (!skipFrame)
        fullSystem->addActiveFrame(img, i);

      delete img;

      if (fullSystem->initFailed || setting_fullResetRequested) {
        if (ii < 250 || setting_fullResetRequested) {
          printf("RESETTING!\n");

          std::vector<IOWrap::Output3DWrapper *> wraps =
              fullSystem->outputWrapper;

          for (IOWrap::Output3DWrapper *ow : wraps)
            ow->reset();

          fullSystem.reset(new FullSystem());
          fullSystem->setGammaFunction(reader->getPhotometricGamma());
          fullSystem->linearizeOperation = (playbackSpeed == 0);

          fullSystem->outputWrapper = wraps;

          setting_fullResetRequested = false;
        }
      }

      if (fullSystem->isLost) {
        printf("LOST!!\n");
        break;
      }
    }
    fullSystem->blockUntilMappingIsFinished();
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    fullSystem->printResult("result.txt");

    int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
    double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) -
                                      reader->getTimestamp(idsToPlay.back()));
    double MilliSecondsTakenSingle =
        1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
    double MilliSecondsTakenMT =
        sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                              (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    printf("\n======================"
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame (single core); "
           "\n%.2fms per frame (multi core); "
           "\n%.3fx (single core); "
           "\n%.3fx (multi core); "
           "\n======================\n\n",
           numFramesProcessed, numFramesProcessed / numSecondsProcessed,
           MilliSecondsTakenSingle / numFramesProcessed,
           MilliSecondsTakenMT / (float)numFramesProcessed,
           1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
           1000 / (MilliSecondsTakenMT / numSecondsProcessed));
    // fullSystem->printFrameLifetimes();
    if (setting_logStuff) {
      std::ofstream tmlog;
      tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
      tmlog << 1000.0f * (ended - started) /
                   (float)(CLOCKS_PER_SEC * reader->getNumImages())
            << " "
            << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                   (float)reader->getNumImages()
            << "\n";
      tmlog.flush();
      tmlog.close();
    }
  });

  if (viewer)
    viewer->run();

  runthread.join();

  for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
    ow->join();
    delete ow;
  }

  // shutdown exThread
  exThreadKeepRunning = false;
  exThread.join();

  printf("EXIT NOW!\n");
  return 0;
}
} // namespace dso
int main(int argc, char **argv) { dso::Main(argc, argv); }
