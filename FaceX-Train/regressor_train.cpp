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

#include "regressor_train.h"

#include <utility>
#include <iostream>
#include <memory>
#include <algorithm>

#include "utils_train.h"


using namespace std;

RegressorTrain::RegressorTrain(const TrainingParameters &tp)
	: training_parameters(tp)
{
	forests = vector<RFSTrain>(training_parameters.landmark_count, RFSTrain(tp));
	
	int num_leaves = pow(2, training_parameters.depth_trees - 1);
	feat_length = training_parameters.landmark_count * training_parameters.num_trees * num_leaves;

	svm_regressors = new CvSVM[training_parameters.landmark_count * 2];
}

void RegressorTrain::release()
{
	for (int i = 0; i < training_parameters.landmark_count * 2; i++){
		svm_regressors[i].clear();
	}
	// to fix: I can't delete it
	// delete[] svm_regressors;
}

void RegressorTrain::Regress(int index_reg, const vector<cv::Point2d> &mean_shape,
	vector<vector<cv::Point2d>> *targets,
	const vector<DataPoint> & training_data)
{
	cout << "start training random forest.";
	for (int i = 0; i < training_parameters.landmark_count; ++i)
		forests[i].Regress(i, index_reg, mean_shape, targets, training_data);

	cout << "random forest finished, start svm";

	vector<vector<bool>> bin_feats = vector<vector<bool>>(training_data.size(), vector<bool>(feat_length));
	for (int i = 0; i < training_data.size(); ++i)
	{
		for (int j = 0; j < training_parameters.landmark_count; j++)
		{
			forests[j].Apply(j, mean_shape, training_data[i], bin_feats[i]);
		}
	}
	/*
	for (int i = 0; i < training_data.size(); ++i)
	{
	bin_feats[i] = new float[feat_length];
	memset(bin_feats[i], 0, feat_length*sizeof(float));
	for (int j = 0; j < training_parameters.landmark_count; j++)
	{
	forests[j].Apply(j, mean_shape, training_data[i], bin_feats[i]);
	}
	}
	cv::Mat mat_feats(training_data.size(), feat_length, CV_32FC1, bin_feats);
	for (int i = 0; i < training_data.size(); ++i)
	{
	delete[] bin_feats[i];
	}
	delete[] bin_feats;
	}
	*/
	
	GlobalRegress(targets, bin_feats);
}

void RegressorTrain::GlobalRegress(std::vector<std::vector<cv::Point2d>> *targets,
	const vector<vector<bool>> &bin_feats)
{

	cv::Mat mat_feats(bin_feats.size(), bin_feats[0].size(), CV_32FC1);
	for (int i = 0; i < bin_feats.size(); i++)
	{
		for (int j = 0; j < bin_feats[0].size(); j++)
		{
			mat_feats.at<float>(i, j) = bin_feats[i][j];
		}
	}

	cv::SVMParams svm_param;
	svm_param.svm_type = CvSVM::EPS_SVR;
	svm_param.kernel_type = CvSVM::LINEAR;
	svm_param.C = 1;
	svm_param.p = 5e-3;
	svm_param.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 5e-3);

	for (int i = 0; i < training_parameters.landmark_count; i++)
	{
		cv::Mat target_single_x = cv::Mat(mat_feats.rows, 1, CV_32FC1);
		cv::Mat target_single_y = cv::Mat(mat_feats.rows, 1, CV_32FC1);
		for (int j = 0; j < mat_feats.rows; j++)
		{
			target_single_x.at<float>(j, 0) = (*targets)[j][i].x;
			target_single_y.at<float>(j, 0) = (*targets)[j][i].y;
		}
		
		svm_regressors[2 * i].train(mat_feats, target_single_x, cv::Mat(), cv::Mat(), svm_param);
		svm_regressors[2 * i + 1].train(mat_feats, target_single_y, cv::Mat(), cv::Mat(), svm_param);
	}
}

vector<cv::Point2d> RegressorTrain::Apply(const vector<cv::Point2d> &mean_shape, 
	const DataPoint &data) const
{
	vector<cv::Point2d> offset(training_parameters.landmark_count);

	vector<bool> bin_feat = vector<bool>(feat_length);
	for (int i = 0; i < training_parameters.landmark_count; ++i){
		forests[i].Apply(i, mean_shape, data, bin_feat);
	}
	cv::Mat mat_feat(1, feat_length, CV_32FC1);
	for (int i = 0; i < feat_length; i++){
		mat_feat.at<float>(0, i) = bin_feat[i];
	}
	for (int i = 0; i < training_parameters.landmark_count; ++i){
		offset[i].x = svm_regressors[2 * i].predict(mat_feat);
		offset[i].y = svm_regressors[2 * i + 1].predict(mat_feat);
	}
	return offset;
}


void RegressorTrain::write(cv::FileStorage &fs)const
{
	cv::WriteStructContext ws_tis(fs, "", CV_NODE_MAP + CV_NODE_FLOW);
	cv::write(fs, "feat_length", feat_length);
	cv::write(fs, "forests", forests);
	{
		cv::WriteStructContext ws_tis(fs, "svm_regressors", CV_NODE_SEQ + CV_NODE_FLOW);
		for (int i = 0; i < training_parameters.landmark_count * 2; i++)
		{
			svm_regressors[i].write(fs.fs, "");
		}
	}
	
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r)
{
	r.write(fs);
}