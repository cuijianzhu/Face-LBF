/*
The MIT License(MIT)

Copyright(c) 2014 Yichun Shi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef FACE_X_FERN_H_
#define FACE_X_FERN_H_

#include<vector>
#include<utility>

#include<opencv2/core/core.hpp>

#include "utils.h"

struct RTree
{
	void Apply(int num_trees, int num_lm, int index_tree, int index_lm,
		const cv::Mat &image, const std::vector<cv::Point2d> &mean_shape, 
		const std::vector<cv::Point2d> &init_shape, cv::Mat &bin_feat) const;
	void read(const cv::FileNode &fn);

	int depth;
	int num_nodes;
	std::vector<std::pair<cv::Point2d, cv::Point2d>> feats;
	std::vector<double> thresholds;
};

struct RFS
{
	void Apply(int num_lm, int index_lm, const cv::Mat &image, 
		const std::vector<cv::Point2d> &mean_shape, 
		const std::vector<cv::Point2d> &init_shape,
		cv::Mat &bin_feat) const;
	
	void read(const cv::FileNode &fn);

	std::vector<RTree> rtrees;
};

void read(const cv::FileNode& node, RTree& f, const RTree& default_value = RTree());
void read(const cv::FileNode& node, RFS& f, const RFS& default_value = RFS());

#endif