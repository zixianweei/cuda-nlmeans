// Copyright 2024 zixianweei
// Use of this source code is governed by a MIT-style license that can be
// found in the LICENSE file.

#include "utilities.h"
#include <opencv2/imgproc.hpp>

void add_noise(float sigma, const cv::Mat& image, cv::Mat& image_noise)
{
	cv::Mat image_f;
	image.convertTo(image_f, CV_32FC1);
	cv::Mat noise_f = cv::Mat(image.size(), CV_32FC1);
	cv::randn(noise_f, 0, sigma);
	cv::Mat image_noise_f = image_f + noise_f;
	image_noise_f.convertTo(image_noise, CV_8UC1);
}

void run_nlmeans_cpu(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h, float sigma)
{
	int patch_radius = patch_size >> 1;
	int block_radius = block_size >> 1;
	int pad_size = block_radius + patch_radius;
	int block_cnt = block_size * block_size;
	int patch_cnt = patch_size * patch_size;
	float ph_div = 1.0 / (float)(patch_cnt);
	int center_pos = block_cnt / 2 + 1;

	cv::Mat input_pad;
	cv::copyMakeBorder(input, input_pad, pad_size, pad_size, pad_size, pad_size, cv::BORDER_DEFAULT);

	float *lut = new float[65536];
	float stddev = sigma == 0.0 ? h : sigma;
	float coeff = -1.0 / (h * h);

	for (int i = 0; i < 65536; ++i) {
		float v = std::exp(std::max(i - 2.0 * stddev * stddev, 0.0) * coeff);
		if (v < 0.001) {
			lut[i] = 0.0;
		} else {
			lut[i] = v;
		}
	}

	for (int row = 0; row < input.rows; ++row) {
		for (int col = 0; col < input.cols; ++col) {
			int *weight = new int[block_cnt];
			float *norm_weight = new float[block_cnt];
			
			float total_weight = 0.0;
			int cnt = 0;

			for (int r = 0; r < block_size; ++r) {
				for (int c = 0; c < block_size; ++c) {
					int ph_weight = 0;

					for (int j = 0; j < patch_size; ++j) {
						for (int i = 0; i < patch_size; ++i) {
							float s = input_pad.data[input_pad.step * (block_radius + row + j) + block_radius + col + i];
							float t = input_pad.data[input_pad.step * (row + r + j) + col + c + i];
							ph_weight += (s - t) * (s - t);
						}
					}

					int w = ph_weight * ph_div;
					weight[cnt++] = w;
					total_weight += lut[w];
				}
			}

			if (total_weight == 0.0) {
				for (int i = 0; i < block_cnt; ++i) {
					norm_weight[i] = 0.0;
				}
				norm_weight[center_pos] = 1.0;
			} else {
				float norm_coeff = 1.0 / (float)total_weight;
				for (int i = 0; i < block_cnt; ++i) {
					norm_weight[i] = lut[weight[i]] * norm_coeff;
				}
			}

			cnt = 0;
			float v = 0.0;
			for (int r = 0; r < block_size; ++r) {
				for (int c = 0; c < block_size; ++c) {
					float pixel = input_pad.data[input_pad.step * (row + patch_radius + r) + col + patch_radius + c];
					v += norm_weight[cnt++] * pixel;
				}
			}

			output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v);

			delete[] weight;
			delete[] norm_weight;
		}
	}

	delete[] lut;
}

void run_nlmeans_cpu_omp(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h, float sigma)
{
	int patch_radius = patch_size >> 1;
	int block_radius = block_size >> 1;
	int pad_size = block_radius + patch_radius;
	int block_cnt = block_size * block_size;
	int patch_cnt = patch_size * patch_size;
	float ph_div = 1.0 / (float)(patch_cnt);
	int center_pos = block_cnt / 2 + 1;

	cv::Mat input_pad;
	cv::copyMakeBorder(input, input_pad, pad_size, pad_size, pad_size, pad_size, cv::BORDER_DEFAULT);

	float *lut = new float[65536];
	float stddev = sigma == 0.0 ? h : sigma;
	float coeff = -1.0 / (h * h);

	for (int i = 0; i < 65536; ++i) {
		float v = std::exp(std::max(i - 2.0 * stddev * stddev, 0.0) * coeff);
		if (v < 0.001) {
			lut[i] = 0.0;
		} else {
			lut[i] = v;
		}
	}

#pragma omp parallel for
	for (int row = 0; row < input.rows; ++row) {
		for (int col = 0; col < input.cols; ++col) {
			int *weight = new int[block_cnt];
			float *norm_weight = new float[block_cnt];

			float total_weight = 0.0;
			int cnt = 0;

			for (int r = 0; r < block_size; ++r) {
				for (int c = 0; c < block_size; ++c) {
					int ph_weight = 0;

					for (int j = 0; j < patch_size; ++j) {
						for (int i = 0; i < patch_size; ++i) {
							float s = input_pad.data[input_pad.step * (block_radius + row + j) + block_radius + col + i];
							float t = input_pad.data[input_pad.step * (row + r + j) + col + c + i];
							ph_weight += (s - t) * (s - t);
						}
					}

					int w = ph_weight * ph_div;
					weight[cnt++] = w;
					total_weight += lut[w];
				}
			}

			if (total_weight == 0.0) {
				for (int i = 0; i < block_cnt; ++i) {
					norm_weight[i] = 0.0;
				}
				norm_weight[center_pos] = 1.0;
			} else {
				float norm_coeff = 1.0 / (float)total_weight;
				for (int i = 0; i < block_cnt; ++i) {
					norm_weight[i] = lut[weight[i]] * norm_coeff;
				}
			}

			cnt = 0;
			float v = 0.0;
			for (int r = 0; r < block_size; ++r) {
				for (int c = 0; c < block_size; ++c) {
					float pixel = input_pad.data[input_pad.step * (row + patch_radius + r) + col + patch_radius + c];
					v += norm_weight[cnt++] * pixel;
				}
			}

			output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v);

			delete[] weight;
			delete[] norm_weight;
		}
	}

	delete[] lut;
}

double calc_psnr(const cv::Mat& i1, const cv::Mat& i2)
{
	cv::Mat s1;
	cv::absdiff(i1, i2, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);

	cv::Scalar s = cv::sum(s1);

	double sse = s.val[0] + s.val[1] + s.val[2];

	if (sse <= 1e-10) {
		return 0;
	}
	double  mse = sse / (double)(i1.channels() * i1.total());
	double psnr = 10.0*log10((255 * 255) / mse);
	return psnr;
}

double calc_ssim(const cv::Mat& i1, const cv::Mat& i2)
{
	static const double C1 = 6.5025, C2 = 58.5225;

	cv::Mat I1, I2;
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);

	cv::Mat I2_2 = I2.mul(I2);  
	cv::Mat I1_2 = I1.mul(I1);  
	cv::Mat I1_I2 = I1.mul(I2); 				

	cv::Mat mu1, mu2;
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;

	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	
	cv::Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);

	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);
	cv::Scalar mssim = cv::mean(ssim_map);
	
	double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
	return ssim * 3;
}