/*
*  Copyright 2017 Pedro Proenca <p.proenca@surrey.ac.uk> (University of Surrey)
*
*  TODO: Out of Bounds
*/

#include "Dataset.h"
#include "OMGUtilities.hpp"
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <cpd/nonrigid.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace OMG;
using namespace Eigen;



int main(int argc, char ** argv)
{

	//viz::Viz3d viewer = viz::Viz3d( "Point Cloud" );
	//Mat cloudMat(cloud_reg.rows(),cloud_reg.cols(),CV_32F);
	//cout << "rows:" << cloud_reg.rows() << "cols:" << cloud_reg.cols() << "\n";
	//eigen2cv(cloud_reg,cloudMat);
	//imshow("?",cloudMat);
	//viz::WCloud cloud(cloudMat,cloudMat);
	//viewer.showWidget("Cloud",cloud);

	cout<<"Nr Threads: "<<nbThreads()<<endl;

	string dataset_name = "VERA";
	string sequence = "vera";
    //string dataset_name = "ICL_NUIM";
    //string sequence = "living_room_traj0_frei_png_short";

	int MAX_NR_FUSION_FRAMES = 5;
	float consistency_threshold = 9;
	if (argc>2){
		MAX_NR_FUSION_FRAMES = atoi(argv[1]);
		consistency_threshold = atof(argv[2]);
		dataset_name = argv[3];
		sequence = argv[4];
	}/*else{
		cout<<"Insert dataset name:";
		cin>>dataset_name;
		cout<<"Insert sequence name:";
		cin>>sequence;
		cout<<"Insert maximum number of frames (default 5): ";
		cin>>MAX_NR_FUSION_FRAMES;
		cout<<"Insert consistency criteria: (default 9): ";
		cin>>consistency_threshold;
	}*/

	stringstream string_buff;
	string data_path = "../../Data/";
	int dataset_type = 0;

	// Get sequence's file structure
	string_buff<<data_path<<dataset_name<<"/"<<sequence;
	Dataset dataset(string_buff.str(),dataset_name);
	dataset.load_and_align_pose_gt(string_buff.str());

	// Get intrinsics
	Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
	string_buff<<"/calib_params.xml";
	loadCalibParameters(string_buff.str(), K_rgb, dist_coeffs_rgb, K_ir, dist_coeffs_ir, R_stereo, t_stereo);
	float fx = K_rgb.at<double>(0,0); float fy = K_rgb.at<double>(1,1); float cx = K_rgb.at<double>(0,2);
	float cy = K_rgb.at<double>(1,2);

	// Read frame 1 to allocate and get dimension
	Mat rgb_img = imread(dataset.rgb_files[0], IMREAD_COLOR);
	Mat d_img = imread(dataset.d_files[0], IMREAD_ANYDEPTH);
	int width = rgb_img.cols;
	int height = rgb_img.rows;
	/* -------------------- Allocation & pre-computation -------------------- */ 
	// Used for backprojection
	Mat_<float> X_pre(height,width);
	Mat_<float> Y_pre(height,width); 
	// Incidence angle used to convert depth to range
	Mat_<float> cos_alpha(height,width); 
	Mat_<float> cos_alpha_sqr(height,width);
	for (int r=0;r<height; r++){
		for (int c=0;c<width; c++){
			float u = c-cx;
			float v = r-cy;
			float D = sqrt(u*u+v*v);
			X_pre.at<float>(r,c) = u/fx; Y_pre.at<float>(r,c) = v/fy;
			cos_alpha.at<float>(r,c) = cos(atan(D/fx));
		}
	}
	cos_alpha_sqr = cos_alpha.mul(cos_alpha);

	int mxn = height*width;
	float * point_cloud_array = (float*)malloc(3*mxn*sizeof(float));
	float * point_range_array = (float*)malloc(mxn*sizeof(float));
	float * point_var_array = (float*)malloc(mxn*sizeof(float));
	float * point_var_reg_array = (float*)malloc(MAX_NR_FUSION_FRAMES*mxn*sizeof(float));
	MatrixXf cloud_reg(MAX_NR_FUSION_FRAMES*mxn,3);
	MatrixXf cloud_frame(mxn,3);
	ArrayXf range_frame(mxn,1);
	ArrayXf range_reg(MAX_NR_FUSION_FRAMES*mxn,1);
	ArrayXf range_reg_sqr(MAX_NR_FUSION_FRAMES*mxn,1);
	Mat_<float> d_sensor_model_uncertainty(height,width);
	Mat_<float> d_conv(height,width);
	Mat_<float> d_conv_uncertainty(height,width); // output
	Mat_<float> d_sqr(height,width);
	Mat_<float> normalizer(height,width); // temporary
	Mat_<float> range_map(height,width);
	Mat_<float> X(height,width);
	Mat_<float> Y(height,width);
	Mat_<float> range_cov_map(height,width);;
	Mat_<float> range_OMG_map(height,width);
	Mat_<float> range_OMG_var_map(height,width);
	Mat_<float> depth_OMG_map(height,width);
	Mat_<float> depth_OMG_var_map(height,width);
	Mat_<int> fusion_counter(height,width);
	Mat_<Vec3f> depth_map_jet(height,width);
	Mat_<float> norm_weights(height,width);
	ArrayXi U_reg(MAX_NR_FUSION_FRAMES*mxn,1);
	ArrayXi V_reg(MAX_NR_FUSION_FRAMES*mxn,1);
	int nr_points_acc = 0;
	int * frame_ptr = (int*)malloc(MAX_NR_FUSION_FRAMES*sizeof(int));
	for(int j=0; j<MAX_NR_FUSION_FRAMES; j++)
		frame_ptr[j] = 0;

	Matrix4d T_old;

	/* ---------------------------------------------------------------------- */ 
	namedWindow("Fused Depth");
	moveWindow("Fused Depth",0,0);
	namedWindow("Raw Depth");
    moveWindow("Raw Depth",0,600);
	namedWindow("Fused Depth Uncertainty");
    moveWindow("Fused Depth Uncertainty",600,0);

	for(int i=0; i<dataset.rgb_tstamps.size(); i++){

		// Read frame i
		rgb_img = imread(dataset.rgb_files[i], IMREAD_COLOR);
		d_img = imread(dataset.d_files[i], IMREAD_ANYDEPTH);
		d_img.convertTo(d_img, CV_32F);
		//KP: why is depth divided by 5? because of conversion to CV_32F
		//d_img = d_img/5;

		// Removing depth image margins due to extreme noise as suggested in the paper
		// KP: that's probably bad idea due to Kinect's person detection
		if (dataset_name.compare("ICL_NUIM")==0){
			d_img(Range(0,5),Range(0,width)) = 0.0f;
			d_img(Range(height-5,height),Range(0,width)) = 0.0f;
			d_img(Range(0,height),Range(0,5)) = 0.0f;
			d_img(Range(0,height),Range(width-5,width)) = 0.0f;
		}

		double t1 = getTickCount();
		// Kinect 1 sensor model uncertainty according to Khoshelham and Elberink:
		// sigma_z = 1.425*10^(-6)*Z^2

		//KP: hypothesis: Kinect v2 uncertaintity is not quadratic with the distance, because it is based of ToF
		//KP: furthere research is needed to find exactly what the model should be
		pow(d_img.mul(d_img)*0.000001425,2,d_sensor_model_uncertainty);

		/* -------------------- Convolution of Gaussian Mixtures -------------------- */
		Mat kernel = (Mat_<float>(3,3) << 1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f); // kernel
		// Obtain normalizing factor S
		min(d_img, 1, normalizer);
		filter2D(normalizer,normalizer,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
		// Obtain MG mea+n
		filter2D(d_img,d_conv,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
		divide(d_conv,normalizer,d_conv,1);
		// Obtain MG var
		pow(d_img,2,d_sqr);
		filter2D(d_sqr+d_sensor_model_uncertainty,d_conv_uncertainty,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
		divide(d_conv_uncertainty,normalizer,d_conv_uncertainty,1);
		d_conv_uncertainty = d_conv_uncertainty - d_conv.mul(d_conv);

		/* --------------------- Convert depth to range --------------------- */
		divide(d_conv,cos_alpha,range_map,1);
		divide(d_conv_uncertainty,cos_alpha_sqr,range_cov_map,1);
		/* ------------------------------------------------------------------ */
		// Backproject to point cloud
		X = X_pre.mul(d_conv); Y = Y_pre.mul(d_conv);
		// Do housecleaning to remove Zeros or NaNs
		int nr_valid_pts = remove_invalid_points(X, Y, d_conv, range_map, range_cov_map, point_cloud_array, point_range_array, point_var_array);
		cloud_frame = Map<MatrixXf>(point_cloud_array, mxn, 3);
		range_frame = Map<ArrayXf>(point_range_array, mxn, 1); 

		/* -------------------- Remove old measurements -------------------- */
		// Just shifting data
		int offset  = frame_ptr[1];
		nr_points_acc -= offset;
		cloud_reg.block(0,0,nr_points_acc,3) = cloud_reg.block(offset,0,nr_points_acc,3);
		memcpy(point_var_reg_array, &point_var_reg_array[offset], sizeof(float)*nr_points_acc);
		range_reg.block(0,0,nr_points_acc,1) = range_reg.block(offset,0,nr_points_acc,1);
		for(int j=0;j<MAX_NR_FUSION_FRAMES;j++)
			frame_ptr[j]-=offset;

		/* -------------------- Transform point cloud -------------------- */ 
		// Using gt pose for demonstration purposes
		// Get absolute pose
		double t_new[3] = {1000*dataset.gt_poses[i][0], 1000*dataset.gt_poses[i][1], 1000*dataset.gt_poses[i][2]};
		double q_new[4] = {dataset.gt_poses[i][3], dataset.gt_poses[i][4], dataset.gt_poses[i][5], dataset.gt_poses[i][6]};

		//KP: rotation matrix
		Matrix3d R_new;
		convertQuaternion2SO3(q_new,R_new);

		//KP: transformation matrix (with rotation)
		Matrix4d T_new = Matrix4d::Identity();
		T_new.block(0,0,3,3) = R_new;
		T_new(0,3) = t_new[0]; T_new(1,3) = t_new[1]; T_new(2,3) = t_new[2];

		// Compute local transformation
		Matrix4d	dT = (T_old.inverse()*T_new).inverse();
		T_old = T_new;

		MatrixXf dR = dT.transpose().cast<float>().block(0,0,3,3);
		float tx = dT(0,3); float ty = dT(1,3); float tz = dT(2,3);

		if(nr_points_acc>0){
			// Rotate
			cloud_reg.block(0,0,nr_points_acc,3) = cloud_reg.block(0,0,nr_points_acc,3)*dR;
			// Translate
			cloud_reg.col(0) = cloud_reg.col(0).array()+tx; 
			cloud_reg.col(1) = cloud_reg.col(1).array()+ty; 
			cloud_reg.col(2) = cloud_reg.col(2).array()+tz; 
		}

		/* --------------------------- FUSION --------------------------------- */
		// TODO: Remove pixels out-of-bounds, Check pixel consistency
		// Initialize images with current frame
		norm_weights = 0.0f;
		range_OMG_map = 0.0f;
		range_OMG_var_map = 0.0f;
		range_OMG_var_map = 0.0f;
		fusion_counter = 1.0f;
		norm_weights += 1/range_cov_map;
		range_OMG_map += range_map.mul(norm_weights);
		range_OMG_var_map += norm_weights.mul((range_cov_map + range_map.mul(range_map)));

		range_reg_sqr = range_reg*range_reg;
		// Projection
		U_reg = (fx*cloud_reg.col(0).array()/cloud_reg.col(2).array() + cx).template cast<int>();
		V_reg = (fy*cloud_reg.col(1).array()/cloud_reg.col(2).array() + cy).template cast<int>();

		for(int pt=nr_points_acc-1; pt>-1; pt--){
			int u_p = U_reg(pt); int v_p = V_reg(pt); float r_p = range_reg(pt);
			float r_var = point_var_reg_array[pt];
			if(r_p>0 && u_p>-1 && v_p>-1 && u_p < 640 && v_p < 480){
				float r_var_inv = 1/r_var;
				float r_curr = range_OMG_map.at<float>(v_p,u_p)/norm_weights.at<float>(v_p,u_p);
				float r_var_curr = range_OMG_var_map.at<float>(v_p,u_p)/norm_weights.at<float>(v_p,u_p) - r_curr*r_curr;
				if (fusion_counter.at<int>(v_p,u_p)<3 || pow(r_p-r_curr,2)<consistency_threshold*r_var_curr){
					norm_weights.at<float>(v_p,u_p) += r_var_inv;
					range_OMG_map.at<float>(v_p,u_p) += r_p*r_var_inv;
					range_OMG_var_map.at<float>(v_p,u_p) += r_var_inv*(r_var + range_reg_sqr(pt));
					fusion_counter.at<int>(v_p,u_p) += 1; 
				}
			}
		}

		divide(range_OMG_map,norm_weights,range_OMG_map,1);
		divide(range_OMG_var_map,norm_weights,range_OMG_var_map,1);
		range_OMG_var_map -= range_OMG_map.mul(range_OMG_map);

		// Convert range to depth
		depth_OMG_map = range_OMG_map.mul(cos_alpha);
		depth_OMG_var_map =  range_OMG_var_map.mul(cos_alpha_sqr);

		/* ----------------------- Update point cloud with new measurements ----------------------- */

		cloud_reg.block(nr_points_acc,0,nr_valid_pts,3) = cloud_frame.block(0,0,nr_valid_pts,3);
		memcpy(&point_var_reg_array[nr_points_acc], point_var_array, sizeof(float)*nr_valid_pts);
		range_reg.block(nr_points_acc,0,nr_valid_pts,1) = range_frame.block(0,0,nr_valid_pts,1);

		// Update pointers
		for(int j=0; j<MAX_NR_FUSION_FRAMES-1;j++)
			frame_ptr[j] = frame_ptr[j+1];

		frame_ptr[MAX_NR_FUSION_FRAMES-1] = nr_points_acc;
		nr_points_acc += nr_valid_pts;
		double t2 = getTickCount();
		double time_elapsed = (t2-t1)/(double)getTickFrequency();
		//cout<<"Time elapsed: "<<time_elapsed<<" s"<< endl;

		/* ------------------------------------- Show Images -------------------------------------- */

        double min_d, max_d;

        //KP: last param updated from 4000 to 50000, because we are not dividing d_img by 5
		convertGray2jet(d_img, depth_map_jet,500,50000);
		//imshow("Raw Depth",depth_map_jet);
		//waitKey(1);

		convertGray2jet(depth_OMG_map, depth_map_jet,500,4000);
        minMaxIdx(depth_OMG_map, &min_d,&max_d);
        //cout << "Min:" << min_d << " Max:" << max_d << " | ";
        //max_d = 65535;
        Mat fusedMap;//(width,height,CV_16UC1);
        //normalize(depth_OMG_map, depth_OMG_map, min_d, max_d, NORM_MINMAX, CV_16UC1);
        depth_OMG_map.convertTo(fusedMap, CV_16UC1);


		minMaxIdx(fusedMap, &min_d,&max_d);
		cout << "Min:" << min_d << " Max:" << max_d << " | \n";
		dilate(fusedMap, fusedMap, Mat(), Point(-1, -1), 1, 1, 1);
		imshow("Fused Depth",fusedMap);
		//imwrite(dataset.d_files[i].replace(dataset.d_files[i].find("depth"), 5, "fused"), fusedMap);
		//waitKey(1);

		minMaxIdx(depth_OMG_var_map, &min_d,&max_d);max_d = 65535;
		Mat adjMap;
		convertScaleAbs(depth_OMG_var_map, adjMap, 255/max_d);
		//imshow("Fused Depth Uncertainty",adjMap);
		//waitKey(1);

	}
	return 0;
}

