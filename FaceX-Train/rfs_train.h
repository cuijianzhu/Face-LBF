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

#ifndef FACE_X_FRS_TRAIN_H_
#define FACE_X_FRS_TRAIN_H_

#include<vector>
#include<utility>

#include<opencv2/core/core.hpp>

#include "utils_train.h"


struct RTreeTrain{
	RTreeTrain(const TrainingParameters &tp);
	void Regress(int index_lm, int index_reg, const std::vector<cv::Point2d> &mean_shape,
		std::vector<std::vector<cv::Point2d>> *targets,
		const std::vector<DataPoint> & training_data,
		const std::pair<int, int> &data_range);
	void Apply(int index_tree, int index_lm, const std::vector<cv::Point2d> &mean_shape,
		const DataPoint &data, bool *bin_feat) const;

private:
	int depth;
	int num_nodes;
	std::vector<std::pair<cv::Point2d,cv::Point2d>> feats;	// the final selected features for each node
	std::vector<double> thresholds;
	const TrainingParameters &training_parameters;
};

struct RFSTrain
{
	RFSTrain(const TrainingParameters &tp);
	void Regress(int index_lm, int index_reg, 
		const std::vector<cv::Point2d> &mean_shape,
		std::vector<std::vector<cv::Point2d>> *targets,
		const std::vector<DataPoint> & training_data);
	void Apply(int index_lm, const std::vector<cv::Point2d> &mean_shape,
		const DataPoint &data, bool *bin_feat) const;

	void write(cv::FileStorage &fs)const;

private:
	std::vector<RTreeTrain> rtrees;
	const TrainingParameters &training_parameters;
};

void write(cv::FileStorage& fs, const std::string&, const RFSTrain& f);

#endif