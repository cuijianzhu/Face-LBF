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
	num_trees_all = training_parameters.landmark_count * training_parameters.num_trees;

	glb_weight = vector<vector<double>>(feat_length, vector<double>(training_parameters.landmark_count * 2));
}

void RegressorTrain::release()
{	
}

void RegressorTrain::Regress(int index_reg, const vector<cv::Point2d> &mean_shape,
	vector<vector<cv::Point2d>> *targets,
	const vector<DataPoint> & training_data)
{
	for (int i = 0; i < training_parameters.landmark_count; ++i)
	{

		cout << "> training landmark " << i+1;
		forests[i].Regress(i, index_reg, mean_shape, targets, training_data);
		cout << endl;
	}

	feature_node** bin_feat_nodes = new feature_node*[training_data.size()];
	for (int i = 0; i < training_data.size(); ++i)
	{
		bin_feat_nodes[i] = new feature_node[num_trees_all + 1];
	}
	for (int i = 0; i < training_data.size(); ++i)
	{
		for (int j = 0; j < training_parameters.landmark_count; j++)
		{
			forests[j].Apply(j, mean_shape, training_data[i], bin_feat_nodes[i]);
		}
		bin_feat_nodes[i][num_trees_all].index = -1;
		bin_feat_nodes[i][num_trees_all].value = -1;
	}
	
	GlobalRegress(targets, bin_feat_nodes);
}

void RegressorTrain::GlobalRegress(std::vector<std::vector<cv::Point2d>> *targets,
	feature_node** bin_feat_nodes)
{
	int num_samples = targets->size();
	int num_lm = (*targets)[0].size();
	problem* prob = new problem;
	prob->l = num_samples;
	prob->n = feat_length;
	prob->x = bin_feat_nodes;
	prob->bias = 0;

	parameter* param = new parameter;
	param->solver_type = L2R_L2LOSS_SVR_DUAL;
	param->C = 1.0 / num_samples;
	param->p = 0;

	double** yy = new double*[num_lm * 2];

	for (int i = 0; i < num_lm * 2; i++){
		yy[i] = new double[num_samples];
	}
	for (int i = 0; i < num_lm; i++){
		for (int j = 0; j<num_samples; j++){
			yy[2 * i][j] = (*targets)[j][i].x;
			yy[2 * i + 1][j] = (*targets)[j][i].y;
		}
	}
	// check the features
	/*for (int i = 100; i < 105; i++)
	{
		for (int j = 0; j < num_trees_all; j++)
		{
			cout << "("<< bin_feat_nodes[i][j].index << "," << bin_feat_nodes[i][j].value<<") ";
		}
		cout << endl;
	}*/
	for (int i = 0; i < num_lm * 2; i++)
	{
		prob->y = yy[i];
		check_parameter(prob, param);
		model* lbfmodel = train(prob, param);
		for (int j = 0; j < feat_length; j++){
			glb_weight[j][i] = lbfmodel->w[j];
		}
		//free_model_content(lbfmodel);
		delete lbfmodel;
	}
	for (int i = 0; i < 2*num_lm; i++){
		delete[] yy[i];
	}
	delete[]  yy;
}

vector<cv::Point2d> RegressorTrain::Apply(const vector<cv::Point2d> &mean_shape, 
	const DataPoint &data) const
{
	vector<cv::Point2d> offset(training_parameters.landmark_count);

	vector<bool> bin_feat = vector<bool>(feat_length);
	for (int i = 0; i < training_parameters.landmark_count; ++i){
		forests[i].Apply(i, mean_shape, data, bin_feat);
	}
	
	for (int i = 0; i < training_parameters.landmark_count; ++i){
		offset[i] = cv::Point2d(0,0);
		for (int j = 0; j < feat_length; ++j){
			offset[i].x += bin_feat[j] * glb_weight[j][2 * i];
			offset[i].y += bin_feat[j] * glb_weight[j][2 * i + 1];
		}
	}
	return offset;
}


void RegressorTrain::write(cv::FileStorage &fs)const
{
	cv::WriteStructContext ws_tis(fs, "", CV_NODE_MAP + CV_NODE_FLOW);
	cv::write(fs, "feat_length", feat_length);
	cv::write(fs, "forests", forests);

	// the linear models will be saved seperately
	{
		cv::WriteStructContext ws_tis(fs, "glb_weight", CV_NODE_SEQ + CV_NODE_FLOW);
		for (int i = 0; i < glb_weight.size(); ++i){
			cv::write(fs, "", glb_weight[i]);
		}
	}
	vector<double>();
	
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r)
{
	r.write(fs);
}