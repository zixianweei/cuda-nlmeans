#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

#include "utilities.h"

template<typename T>
__device__ T* local_malloc(int cnt)
{
	return (T*)malloc(cnt * sizeof(T));
}

__global__ void run(cudaTextureObject_t input_pad, float *output, float *lut, int width, int height, int block_size, int patch_size, float h, float sigma)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < 0 || col > width || row < 0 || row > height) {
		return;
	}

	int patch_radius = patch_size >> 1;
	int block_radius = block_size >> 1;
	int block_cnt = block_size * block_size;
	int patch_cnt = patch_size * patch_size;
	float ph_div = 1.0 / (float)(patch_cnt);
	int center_pos = block_cnt / 2 + 1;

	float total_weight = 0.0;
	int cnt = 0;

	// need dynamic local memory allocation
	// int *weight = (int *)malloc(block_cnt * sizeof(int));
	// float *norm_weight = (float *)malloc(block_cnt * sizeof(float));
	// recommanded maximum block size is 17 -> 17 * 17 = 289
	int weight[289];
	float norm_weight[289];

	for (int r = 0; r < block_size; ++r) {
		for (int c = 0; c < block_size; ++c) {
			int ph_weight = 0;

			for (int j = 0; j < patch_size; ++j) {
				for (int i = 0; i < patch_size; ++i) {
					float s = tex2D<uchar>(input_pad, block_radius + col + i, block_radius + row + j);
					float t = tex2D<uchar>(input_pad, col + c + i, row + r + j);
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
		float norm_coeff = 1.0 / total_weight;
		for (int i = 0; i < block_cnt; ++i) {
			norm_weight[i] = lut[weight[i]] * norm_coeff;
		}
	}

	cnt = 0;
	float v = 0.0;
	for (int r = 0; r < block_size; ++r) {
		for (int c = 0; c < block_size; ++c) {
			float pixel = tex2D<uchar>(input_pad, col + patch_radius + c, row + patch_radius + r);
			v += norm_weight[cnt++] * pixel;
		}
	}

	output[row * width + col] = v;
}

void run_nlmeans_gpu(const cv::Mat& input, cv::Mat& output, int block_size, int patch_size, float h, float sigma)
{
	int patch_radius = patch_size >> 1;
	int block_radius = block_size >> 1;
	int pad_size = block_radius + patch_radius;

	cv::Mat input_pad;
	cv::copyMakeBorder(input, input_pad, pad_size, pad_size, pad_size, pad_size, cv::BORDER_DEFAULT);

	// bind input pad to texture memory
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar>();

	cudaArray *input_pad_d;
	cudaMallocArray(&input_pad_d, &channel_desc, input_pad.cols, input_pad.rows);
	cudaMemcpyToArray(input_pad_d, 0, 0, input_pad.data, input_pad.cols * input_pad.rows * sizeof(uchar), cudaMemcpyHostToDevice);

	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = input_pad_d;

	struct cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = false;

	cudaTextureObject_t tex_obj;
	cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

	// create lut table
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

	// allocate lut device memory and copy to device
	float *lut_d = NULL;
	cudaMalloc(&lut_d, 65536 * sizeof(float));
	cudaMemcpy(lut_d, lut, 65536 * sizeof(float), cudaMemcpyHostToDevice);

	// allocate output host memory
	float *output_h = (float*)malloc(output.cols * output.rows * sizeof(float));
	
	// allocate output device memory
	float *output_d = NULL;
	cudaMalloc(&output_d, output.cols * output.rows * sizeof(float));

	// invoke kernel
	dim3 block(16, 16);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	run<<<grid, block>>>(tex_obj, output_d, lut_d, output.cols, output.rows, block_size, patch_size, h, sigma);

	// kernel launch status check
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(status));
		return;
	}

	// copy result form device to host output_d -> output_h
	cudaMemcpy(output_h, output_d, output.cols * output.rows * sizeof(float), cudaMemcpyDeviceToHost);

	// matrix to Mat format image
	for (int row = 0; row < output.rows; ++row) {
		for (int col = 0; col < output.cols; ++col) {
			float v = output_h[row * output.cols + col];
			output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v);
		}
	}

	// free resource device and host
	cudaFreeArray(input_pad_d);
	cudaFree(lut_d);
	cudaFree(output_d);

	delete[] lut;
	delete[] output_h;

	cudaDestroyTextureObject(tex_obj);
	
	cudaDeviceReset();
}
