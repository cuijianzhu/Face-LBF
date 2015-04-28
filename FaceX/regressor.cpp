/*
The MIT License(MIT)

Copyright(c) 2014 Yang Cao

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

#include "regressor.h"

#include <utility>
#include <iostream>
#include <algorithm>

#include "utils.h"


using namespace std;

vector<cv::Point2d> Regressor::Apply(cv::Mat &image,
	const vector<cv::Point2d> &mean_shape,
	const vector<cv::Point2d> &init_shape) const
{
	vector<cv::Point2d> offset(mean_shape.size());

	vector<bool> bin_feat = vector<bool>(feat_length);
	for (int i = 0; i < mean_shape.size(); ++i){
		forests[i].Apply(mean_shape.size(), i, image, mean_shape, init_shape, bin_feat);
	}
	cv::Mat mat_feat(1, feat_length, CV_32FC1);
	for (int i = 0; i < feat_length; i++){
		mat_feat.at<float>(0, i) = bin_feat[i];
	}
	for (int i = 0; i < mean_shape.size(); ++i){
		offset[i] = cv::Point2d(0, 0);
		for (int j = 0; j < feat_length; j++){
			offset[i].x += bin_feat[j] * glb_weight[j][2 * i];
			offset[i].y += bin_feat[j] * glb_weight[j][2 * i + 1];
		}
		cout << offset[i].x << " ";
	}
	cout << endl;
	return offset;
}
void Regressor::read(const cv::FileNode &fn)
{
	fn["feat_length"] >> feat_length;

	cv::FileNode forests_node = fn["forests"];
	forests.resize(forests_node.size());
	int idx = 0;
	for (auto it = forests_node.begin(); it != forests_node.end(); ++it)
	{
		*it >> forests[idx];
		idx++;
	}

	cv::FileNode glb_weight_node = fn["glb_weight"];
	glb_weight.resize(glb_weight_node.size());
	idx = 0;
	for (auto it = glb_weight_node.begin(); it != glb_weight_node.end(); ++it)
	{
		*it >> glb_weight[idx];
		idx++;
	}
}

void read(const cv::FileNode& node, Regressor& r, const Regressor& default_value)
{
	if (node.empty())
	{
		r = default_value;
		cout << "One default Regressor. Model file is corrupt!" << endl;
	}
	else
		r.read(node);
}