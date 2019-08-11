#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <vector>

#include "config/ConfigParams.h"
#include "estimators/FundmatrixEstimator.h"


// helper functions
bool readCorrsFromFile(std::string& inputFilePath, std::vector<double>& pointData, unsigned int& numPts)
{
	// read data from from file
	std::ifstream infile(inputFilePath.c_str());
	if (!infile.is_open())
	{
		std::cerr << "Error opening input points file: " << inputFilePath << std::endl;
		return false;
	}
	infile >> numPts;
	pointData.resize(6*numPts);
	for (unsigned int i = 0; i < numPts; ++i)
	{
		infile >> pointData[6*i] >> pointData[6*i+1] >> pointData[6*i+3] >> pointData[6*i+4];
		pointData[6*i+2] = 1.0;
		pointData[6*i+5] = 1.0;
	}
	infile.close();
	return true;
}


// ---------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// check command line args
	if (argc < 3)
	{
		std::cerr << "Usage: RunSingleTest <estimation problem> <config file>" << std::endl;
		std::cerr << "\t<estimation problem>: 0 (fundamental matrix), 1 (homography)" << std::endl;
		std::cerr << "\t<config file>: full path to configuration file" << std::endl;
		return(EXIT_FAILURE);
	}
	int estimation_problem = atoi(argv[1]);
	std::string cfg_file_path = argv[2];

	// seed random number generator
	srand((unsigned int)time(NULL));


	// ------------------------------------------------------------------------
	// initialize the fundamental matrix estimation problem
	ConfigParamsFund cfg;
	if ( !cfg.initParamsFromConfigFile(cfg_file_path) )
	{
		std::cerr << "Error during initialization" << std::endl;
		return(EXIT_FAILURE);
	}
	FundMatrixEstimator* fund = new FundMatrixEstimator;
	fund->initParamsUSAC(cfg);

	// read in input data points
	std::vector<double> point_data;
	if ( !readCorrsFromFile(cfg.fund.inputFilePath, point_data, cfg.common.numDataPoints) )
	{
		return(EXIT_FAILURE);
	}

	// set up the fundamental matrix estimation problem

	fund->initDataUSAC(cfg);
	fund->initProblem(cfg, &point_data[0]);
	if (!fund->solve())
	{
		return(EXIT_FAILURE);
	}

	// write out results
	size_t pos = (cfg.fund.inputFilePath).find_last_of("/\\");
	std::string working_dir = (cfg.fund.inputFilePath).substr(0, pos + 1);
	std::ofstream outmodel((working_dir + "F.txt").c_str());
	for (unsigned int i = 0; i < 3; ++i)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{
			outmodel << fund->final_model_params_[3*i+j] << " ";
		}
	}
	outmodel.close();
	std::ofstream outinliers((working_dir + "inliers.txt").c_str());
	for (unsigned int i = 0; i < cfg.common.numDataPoints; ++i)
	{
		outinliers << fund->usac_results_.inlier_flags_[i] << std::endl;
	}
	outinliers.close();

	// clean up
	point_data.clear();
	fund->cleanupProblem();
	delete fund;

	return(EXIT_SUCCESS);
}
