/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2014  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#include "rfs_train.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>

using namespace std;

RTreeTrain::RTreeTrain(const TrainingParameters &tp) : training_parameters(tp)
{
	depth = training_parameters.depth_trees;
	num_nodes = pow(2, depth) - 1;
	feats = vector<std::pair<cv::Point2d, cv::Point2d>>(num_nodes);
	thresholds = vector<double>(num_nodes);
}

void RTreeTrain::Regress(int index_lm, int index_reg, const std::vector<cv::Point2d> &mean_shape,
	std::vector<std::vector<cv::Point2d>> *targets,
	const std::vector<DataPoint> & training_data,
	const std::pair<int, int> &data_range)
{
	int is = data_range.first;
	int ie = data_range.second;

	//the indices of samples categorized to each node
	vector<vector<int>> data_node = vector<vector<int>>(num_nodes);
	
	//init the first node with all samples
	data_node[0].resize(ie-is);
	for (int i = 0; i < ie - is; i++) data_node[0][i] = is + i;
	
	//split every node
	int num_nodes_split = (num_nodes - 1) / 2;
	for (int idx_node = 0; idx_node < num_nodes_split; ++idx_node)
	{
		if (data_node[idx_node].empty())
		{
			thresholds[idx_node] = 0;
			feats[idx_node] = pair<cv::Point2d, cv::Point2d>(cv::Point2d(0,0), cv::Point2d(0,0));
			continue;
		}
		//get the random features and their values in each sample
		cv::Mat feats_val(data_node[idx_node].size(), training_parameters.num_feats, CV_64FC1);
		vector<cv::Point2d> feats_first = vector<cv::Point2d>(training_parameters.num_feats);
		vector<cv::Point2d> feats_second = vector<cv::Point2d>(training_parameters.num_feats);
		for (int i = 0; i < training_parameters.num_feats; ++i)
		{
			feats_first[i].x = cv::theRNG().uniform(-training_parameters.radius_feats[index_reg],
				training_parameters.radius_feats[index_reg]);
			feats_first[i].y = cv::theRNG().uniform(-training_parameters.radius_feats[index_reg],
				training_parameters.radius_feats[index_reg]);
			feats_second[i].x = cv::theRNG().uniform(-training_parameters.radius_feats[index_reg],
				training_parameters.radius_feats[index_reg]);
			feats_second[i].y = cv::theRNG().uniform(-training_parameters.radius_feats[index_reg],
				training_parameters.radius_feats[index_reg]);
		}
		for (int itr_data = 0; itr_data < data_node[idx_node].size(); ++itr_data)
		{
			int i = data_node[idx_node][itr_data];
			Transform t = Procrustes(training_data[i].init_shape, mean_shape);
			vector<cv::Point2d> offsets_first(training_parameters.num_feats);
			vector<cv::Point2d> offsets_second(training_parameters.num_feats);
			for (int j = 0; j < training_parameters.num_feats; ++j)
			{
				offsets_first[j] = feats_first[j];
				offsets_second[j] = feats_second[j];
			}
			t.Apply(&offsets_first, false);
			t.Apply(&offsets_second, false);

			for (int j = 0; j < training_parameters.num_feats; ++j)
			{

				cv::Point feat_pos_first = training_data[i].init_shape[index_lm] + offsets_first[j];
				cv::Point feat_pos_second = training_data[i].init_shape[index_lm] + offsets_second[j];
				if (feat_pos_first.inside(cv::Rect(0, 0, training_data[i].image.cols, training_data[i].image.rows))
					&& feat_pos_second.inside(cv::Rect(0, 0, training_data[i].image.cols, training_data[i].image.rows)))
				{
					double s = training_data[i].image.at<uchar>(feat_pos_first)
						-training_data[i].image.at<uchar>(feat_pos_second);
					feats_val.at<double>(itr_data, j) =
						training_data[i].image.at<uchar>(feat_pos_first)
						- training_data[i].image.at<uchar>(feat_pos_second);
				}
				else
					feats_val.at<double>(itr_data, j) = 0;
			}
		}

		//calculate the variance
		double mean_x = 0;
		double mean_x2 = 0;
		double mean_y = 0;
		double mean_y2 = 0;
		double var;
		for (int itr_data = 0; itr_data < data_node[idx_node].size(); ++itr_data)
		{
			int i = data_node[idx_node][itr_data];
			mean_x += (*targets)[i][index_lm].x;
			mean_x2 += (*targets)[i][index_lm].x * (*targets)[i][index_lm].x;
			mean_y += (*targets)[i][index_lm].y;
			mean_y2 += (*targets)[i][index_lm].y * (*targets)[i][index_lm].y;
		}
		double base_size = 1.0 / data_node[idx_node].size();
		mean_x *= base_size;
		mean_x2 *= base_size;
		mean_y *= base_size;
		mean_y2 *= base_size;
		var = mean_x2 - mean_x*mean_x + mean_y2 - mean_y*mean_y;
		var = var * data_node[idx_node].size();

		//calculate the variance reduction of each feature and choose the max
		vector<double> var_reduc = vector<double>(training_parameters.num_feats);
		int max_feat = -1;
		double max_var_reduc = - DBL_MAX;
		double max_threshold;
		for (int i = 0; i < training_parameters.num_feats; i++)
		{
			int cnt_l = 0, cnt_r = 0;
			double mean_x_l=0, mean_x2_l=0, mean_y_l=0, mean_y2_l=0;
			double mean_x_r=0, mean_x2_r=0, mean_y_r=0, mean_y2_r=0;
			double thresh = feats_val.at<double>( cv::theRNG().uniform(0, feats_val.rows), i );
			for (int itr_data = 0; itr_data < data_node[idx_node].size(); ++itr_data)
			{
				int j = data_node[idx_node][itr_data];
				if (feats_val.at<double>(itr_data, i) < thresh){
					mean_x_l += (*targets)[j][index_lm].x;
					mean_x2_l += (*targets)[j][index_lm].x * (*targets)[j][index_lm].x;
					mean_y_l += (*targets)[j][index_lm].y;
					mean_y2_l += (*targets)[j][index_lm].y * (*targets)[j][index_lm].y;
					cnt_l++;
				}
				else
				{
					mean_x_r += (*targets)[j][index_lm].x;
					mean_x2_r += (*targets)[j][index_lm].x * (*targets)[j][index_lm].x;
					mean_y_r += (*targets)[j][index_lm].y;
					mean_y2_r += (*targets)[j][index_lm].y * (*targets)[j][index_lm].y;
					cnt_r++;
				}
			}
			double base_l = (cnt_l == 0) ? 0 : (1.0 / cnt_l);
			double base_r = (cnt_r == 0) ? 0 : (1.0 / cnt_r);
			mean_x_l *= base_l;
			mean_x2_l *= base_l;
			mean_y_l *= base_l;
			mean_y2_l *= base_l;
			mean_x_r *= base_r;
			mean_x2_r *= base_r;
			mean_y_r *= base_r;
			mean_y2_r *= base_r;
			double var_split = (mean_x2_l + mean_y2_l - mean_x_l*mean_x_l - mean_y_l*mean_y_l) * cnt_l
				+ (mean_x2_r + mean_y2_r - mean_x_r*mean_x_r - mean_y_r*mean_y_r) * cnt_r;
			if (var - var_split > max_var_reduc)
			{				
				max_feat = i;
				max_var_reduc = var - var_split;
				max_threshold = thresh;
			}
		}
		thresholds[idx_node] = max_threshold;
		feats[idx_node] = pair<cv::Point2d, cv::Point2d>(feats_first[max_feat], feats_second[max_feat]);
		
		// allocate the samples for the children
		// 2*idx_node+1 is the left child
		for (int itr_data = 0; itr_data < data_node[idx_node].size(); ++itr_data)
		{
			int i = data_node[idx_node][itr_data];
			if (feats_val.at<double>(itr_data, max_feat) < max_threshold)
				data_node[2*idx_node+1].push_back(i);
			else
				data_node[2*idx_node+2].push_back(i);
		}
	}
}

void RTreeTrain::Apply(int index_tree, int index_lm, const std::vector<cv::Point2d> &mean_shape,
	const DataPoint &data, float *bin_feat) const
{
	int num_nodes_split = (num_nodes - 1) / 2;
	int idx_node = 0;
	while (idx_node < num_nodes_split){
		// get the feature and go ahead
		double feat;
		Transform t = Procrustes(data.init_shape, mean_shape);
		vector<cv::Point2d> offset_pair(2);
		offset_pair.push_back(feats[idx_node].first);
		offset_pair.push_back(feats[idx_node].second);
		t.Apply(&offset_pair, false);

		cv::Point feat_pos_first = data.init_shape[index_lm] + offset_pair[0];
		cv::Point feat_pos_second = data.init_shape[index_lm] + offset_pair[1];
		if (feat_pos_first.inside(cv::Rect(0, 0, data.image.cols, data.image.rows))
			&& feat_pos_second.inside(cv::Rect(0, 0, data.image.cols, data.image.rows)))
		{
			feat = data.image.at<uchar>(feat_pos_first)
				- data.image.at<uchar>(feat_pos_second);
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
	int bool_index = index_lm * training_parameters.num_trees * num_leaves
		+ index_tree * num_leaves + idx_node - num_nodes;
	if (bool_index > training_parameters.landmark_count * training_parameters.num_trees * num_leaves)
		throw out_of_range("bool index is out of the range of bin_feat during appling the tree");
	bin_feat[bool_index] = 1;
}



RFSTrain::RFSTrain(const TrainingParameters &tp) : training_parameters(tp)
{
	rtrees = vector<RTreeTrain>(training_parameters.num_trees, RTreeTrain(tp));
}

void RFSTrain::Regress(int index_lm, int index_reg, 
	const std::vector<cv::Point2d> &mean_shape,
	std::vector<std::vector<cv::Point2d>> *targets,
	const std::vector<DataPoint> & training_data)
{
	//dispatch the samples for each tree;
	int nt = training_parameters.num_trees;
	vector<pair<int, int>> data_range(nt);
	if (training_data.size() < nt)
		throw invalid_argument("The number of samples should be larger than NumTree");
	for (int i = 0; i < nt; i++)
	{
		//we have no overlap ratio
		int is = training_data.size() / nt * i;
		int ie = training_data.size() / nt * (i + 1);
		if (i == nt - 1) ie = training_data.size();
		data_range[i].first = is;
		data_range[i].second = ie;
	}
	for (int i = 0; i < nt; i++)
	{
		rtrees[i].Regress(index_lm, index_reg, mean_shape, targets, training_data, data_range[i]);
	}
}

void RFSTrain::Apply(int index_lm, const std::vector<cv::Point2d> &mean_shape,
	const DataPoint &data, float *bin_feat) const
{
	for (int i = 0; i < training_parameters.num_trees; i++){
		rtrees[i].Apply(i, index_lm, mean_shape, data, bin_feat);
	}
}

void RFSTrain::write(cv::FileStorage &fs)const
{
	cv::WriteStructContext ws_fern(fs, "", CV_NODE_MAP + CV_NODE_FLOW);
}

void write(cv::FileStorage& fs, const string&, const RFSTrain &f)
{
	f.write(fs);
}