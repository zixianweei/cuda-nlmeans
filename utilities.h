#ifndef UTILITIES_H
#define UTILITIES_H

// Copyright 2022 zixgo
// Use of this source code is governed by a MIT-style license that can be
// found in the LICENSE file.

#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/core.hpp>

void add_noise(float sigma, const cv::Mat& image, cv::Mat& image_noise);

void run_nlmeans_cpu(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h = 10.0, float sigma = 0.0);

void run_nlmeans_cpu_omp(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h = 10.0, float sigma = 0.0);

#ifdef USE_CUDA
void run_nlmeans_gpu(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h = 10.0, float sigma = 0.0);
#endif // USE_CUDA

double calc_psnr(const cv::Mat& i1, const cv::Mat& i2);

double calc_ssim(const cv::Mat& i1, const cv::Mat& i2);

#endif // !UTILITES_H