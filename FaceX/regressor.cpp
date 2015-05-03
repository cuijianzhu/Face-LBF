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
	cv::Mat bin_feat = cv::Mat::zeros(1, feat_length, CV_32FC1);
	cv::Mat mat_offset(1, 2 * mean_shape.size(), CV_32FC1);
	vector<cv::Point2d> offset(mean_shape.size());
	
	// derive binary feature
	long long time = cv::getTickCount();
	for (int i = 0; i < mean_shape.size(); ++i){
		forests[i].Apply(mean_shape.size(), i, image, mean_shape, init_shape, bin_feat);
	}
	cout << "Deriving binary feature uses "
		<< ((cv::getTickCount() - time) / cv::getTickFrequency()) << "s" << endl;
	
	// apply global model
	time = cv::getTickCount();
	mat_offset = bin_feat * w;
	for (int i = 0; i < mean_shape.size(); ++i){
		offset[i].x = mat_offset.at<float>(0,2 * i);
		offset[i].y = mat_offset.at<float>(0, 2 * i + 1);
	}
	cout << "Applying global model uses "
		<< ((cv::getTickCount() - time) / cv::getTickFrequency()) << "s" << endl;
	
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

	cv::FileNode w_node = fn["w"];
	w_node >> w;
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