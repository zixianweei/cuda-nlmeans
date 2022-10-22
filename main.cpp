// MIT License
// 
// Copyright(c) 2022 zixgo
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xphoto.hpp>
#include "utilities.h"

void test_compare_different_methods();
void test_compare_diffreent_platforms();

int main(int argc, char* argv[])
{
	test_compare_different_methods();
	test_compare_diffreent_platforms();

	return 0;
}

// test1 nlmeans vs other denoising methods
void test_compare_different_methods()
{
	// basic parameters
	float sigma = 10;
	float h = 10.f;
	int block_size = 11;
	int patch_size = 7;

	// image paths
	cv::String root_path{ "../../../" };
	cv::String image_name{ "cameraman" };
	cv::String image_format{ ".png" };

	cv::String image_path = root_path + image_name + image_format;
	cv::String image_noise_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + image_format;
	cv::String image_denoise_median_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_median" + image_format;
	cv::String image_denoise_gaussian_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_gaussian" + image_format;
	cv::String image_denoise_bilateral_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_bilateral" + image_format;
	cv::String image_denoise_nlmeans_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_nlmeans" + image_format;
	
	// read source image (without noise)
	cv::Mat image = imread(image_path, cv::IMREAD_GRAYSCALE);
	assert(!image.empty());

	int width = image.cols;
	int height = image.rows;
	int channels = image.channels();

	// print source image information on terminal
	std::cout << "Read image from : " << image_path << std::endl;
	std::cout << "image size : " << std::endl;
	std::cout << "\twidth=" << width << std::endl;
	std::cout << "\theight=" << height << std::endl;
	std::cout << "\tchannel=" << channels << std::endl << std::endl;

	cv::Mat image_noise = cv::Mat::zeros(image.size(), image.type());
	cv::Mat image_denoised = cv::Mat::zeros(image.size(), image.type());

	// add noise	
	std::cout << "Add noise [sigma = " << sigma << "] ... ";
	cv::TickMeter tm;
	tm.start();
	
	add_noise(sigma, image, image_noise);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl << std::endl;

	cv::imwrite(image_noise_path, image_noise);

	// run different denoise algorithm
	// 1. Median Filtering
	std::cout << "Running median filtering ...";
	tm.reset();
	tm.start();

	cv::medianBlur(image_noise, image_denoised, 3);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_median_path, image_denoised);

	// 2. Gaussian Filtering
	std::cout << "Running Gaussian filtering ...";
	tm.reset();
	tm.start();

	cv::GaussianBlur(image_noise, image_denoised, cv::Size(5, 5), sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_gaussian_path, image_denoised);

	// 3. Bilateral Filtering
	std::cout << "Running bilateral filtering ...";
	tm.reset();
	tm.start();

	cv::bilateralFilter(image_noise, image_denoised, 5, sigma, sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_bilateral_path, image_denoised);

	// 4. Non-local Means Filtering
	std::cout << "Running nlmeans filtering ...";
	tm.reset();
	tm.start();

	run_nlmeans_cpu(image_noise, image_denoised, block_size, patch_size, h, sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_nlmeans_path, image_denoised);
}

// test2 performance on different platforms
void test_compare_diffreent_platforms()
{
	// basic parameters
	float sigma = 10;
	float h = 10.f;
	int block_size = 11;
	int patch_size = 7;
	cv::String root_path{ "../../../" };
	cv::String image_name{ "cameraman" };
	cv::String image_format{ ".png" };

	cv::String image_path = root_path + image_name + image_format;
	cv::String image_noise_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + image_format;
	cv::String image_denoise_cpu_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_cpu" + image_format;
	cv::String image_denoise_omp_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_omp" + image_format;
	cv::String image_denoise_gpu_path = root_path + image_name + "_noise" + "_sigma" + std::to_string(int(sigma)) + "_denoised_gpu" + image_format;

	// read source image (without noise)
	cv::Mat image = imread(image_path, cv::IMREAD_GRAYSCALE);
	assert(!image.empty());

	int width = image.cols;
	int height = image.rows;
	int channels = image.channels();

	// print source image information on terminal
	std::cout << "Read image from : " << image_path << std::endl;
	std::cout << "image size : " << std::endl;
	std::cout << "\twidth=" << width << std::endl;
	std::cout << "\theight=" << height << std::endl;
	std::cout << "\tchannel=" << channels << std::endl << std::endl;

	cv::Mat image_noise = cv::Mat::zeros(image.size(), image.type());
	cv::Mat image_denoised = cv::Mat::zeros(image.size(), image.type());

	// add noise	
	std::cout << "Add noise [sigma = " << sigma << "] ... ";
	cv::TickMeter tm;
	tm.start();

	add_noise(sigma, image, image_noise);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl << std::endl;

	cv::imwrite(image_noise_path, image_noise);

	// run on different denoise platform
	// 1. CPU Only
	std::cout << "Running nlmeans on cpu only ...";
	tm.reset();
	tm.start();

	run_nlmeans_cpu(image_noise, image_denoised, block_size, patch_size, h, sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_cpu_path, image_denoised);

	// 2. CPU with OpenMP
	std::cout << "Running nlmeans on cpu with openmp ...";
	tm.reset();
	tm.start();

	run_nlmeans_cpu_omp(image_noise, image_denoised, block_size, patch_size, h, sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_omp_path, image_denoised);

#ifdef USE_CUDA
	// 3. CPU and GPU
	std::cout << "Running nlmeans on cpu and gpu ...";
	tm.reset();
	tm.start();

	run_nlmeans_gpu(image_noise, image_denoised, block_size, patch_size, h, sigma);

	tm.stop();
	std::cout << "done.\n\ttime duration : " << tm.getTimeMilli() << "(msec)" << std::endl;
	std::cout << "\tpsnr : " << calc_psnr(image, image_denoised) << std::endl;
	std::cout << "\tssim : " << calc_ssim(image, image_denoised) << std::endl << std::endl;

	cv::imwrite(image_denoise_gpu_path, image_denoised);
#endif // USE_CUDA
}
