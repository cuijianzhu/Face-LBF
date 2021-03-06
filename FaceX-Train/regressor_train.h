/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2014  Yichun Shi

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

#ifndef FACE_X_REGRESSOR_TRAIN_H_
#define FACE_X_REGRESSOR_TRAIN_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/core/core.hpp>

#include "utils_train.h"
#include "rfs_train.h"
#include "liblinear/linear.h"

class RegressorTrain
{
public:
	cv::Mat w;
	RegressorTrain(const TrainingParameters &tp);
	void Regress(int index_reg, const std::vector<cv::Point2d> &mean_shape,
		std::vector<std::vector<cv::Point2d>> *targets,
		const std::vector<DataPoint> & training_data);
	std::vector<cv::Point2d> Apply(const std::vector<cv::Point2d> &mean_shape, 
		const DataPoint &data) const;

	void write(cv::FileStorage &fs)const;

private:
	int feat_length;
	int num_trees_all;
	std::vector<RFSTrain> forests;
	const TrainingParameters &training_parameters;

	void GlobalRegress(std::vector<std::vector<cv::Point2d>> *targets,
		feature_node** bin_feats);
};

void write(cv::FileStorage& fs, const std::string&, const RegressorTrain& r);

#endif