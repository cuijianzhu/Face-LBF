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

#include "rfs.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>
#include<opencv2/core/core.hpp>



using namespace std;

void RTree::Apply(int num_trees, int num_lm, int index_tree, int index_lm,
	const cv::Mat &image, const vector<cv::Point2d> &mean_shape,
	const vector<cv::Point2d> &init_shape, vector<bool> &bin_feat) const
{
	int num_nodes_split = (num_nodes - 1) / 2;
	int idx_node = 0;
	while (idx_node < num_nodes_split){
		// get the feature and go ahead
		double feat;
		Transform t = Procrustes(init_shape, mean_shape);
		vector<cv::Point2d> offset_pair(2);
		offset_pair[0] = feats[idx_node].first;
		offset_pair[1] = feats[idx_node].second;
		t.Apply(&offset_pair, false);

		cv::Point feat_pos_first = init_shape[index_lm] + offset_pair[0];
		cv::Point feat_pos_second = init_shape[index_lm] + offset_pair[1];
		if (feat_pos_first.inside(cv::Rect(0, 0, image.cols, image.rows))
			&& feat_pos_second.inside(cv::Rect(0, 0, image.cols, image.rows)))
		{
			feat = image.at<uchar>(feat_pos_first)
				-image.at<uchar>(feat_pos_second);
		}
		else
			feat = 0;

		if (feat < thresholds[idx_node]) idx_node = 2 * idx_node + 1;
		else idx_node = 2 * idx_node + 2;
	}
	if (idx_node >= num_nodes)
		throw out_of_range("idx_node is greater than or equal to num_nodes during appling the tree");
	// calculate the index of this leaf node in the binary feature vector
	int num_leaves = num_nodes - num_nodes_split;
	int bool_index = index_lm * num_trees * num_leaves
		+ index_tree * num_leaves + idx_node - (num_nodes - num_leaves);
	if (bool_index > num_lm * num_trees * num_leaves)
		throw out_of_range("bool index is out of the range of bin_feat during appling the tree");
	bin_feat[bool_index] = 1;
}

void RFS::Apply(int num_lm, int index_lm, const cv::Mat &image,
	const std::vector<cv::Point2d> &mean_shape,
	const std::vector<cv::Point2d> &init_shape,
	std::vector<bool> &bin_feat) const
{
	for (int i = 0; i < rtrees.size(); i++){
		rtrees[i].Apply(rtrees.size(), num_lm, i, index_lm, image, mean_shape, init_shape, bin_feat);
	}
}

void RTree::read(const cv::FileNode &fn)
{
	feats.clear();
	thresholds.clear();
	fn["num_nodes"] >> num_nodes;
	cv::FileNode feats_node = fn["feats"];
	for (auto it = feats_node.begin(); it != feats_node.end(); ++it)
	{
		pair<cv::Point2d, cv::Point2d> feat;
		(*it)["first"] >> feat.first;
		(*it)["second"] >> feat.second;
		feats.push_back(feat);
	}
	cv::FileNode thresholds_node = fn["thresholds"];
	for (auto it = thresholds_node.begin(); it != thresholds_node.end(); ++it)
	{
		double threshold;
		*it >> threshold;
		thresholds.push_back(threshold);
	}
}

void RFS::read(const cv::FileNode &fn)
{
	rtrees.clear();
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		RTree rfs;
		*it >> rfs;
		rtrees.push_back(rfs);
	}
}


void read(const cv::FileNode& node, RTree &f, const RTree& default_value)
{
	if (node.empty())
	{
		f = default_value;
		cout << "! One default Regressor." << endl;
	}
	else
		f.read(node);
}

void read(const cv::FileNode& node, RFS &f, const RFS& default_value)
{
	if (node.empty())
	{
		f = default_value;
		cout << "! One default Regressor." << endl;
	}
	else
		f.read(node);
}