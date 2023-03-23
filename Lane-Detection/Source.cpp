#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/QR>


using namespace std;
using namespace cv;

void cameraCalib(Mat& cameraMatrix, Mat& distCoeffs) {
    int CHECKERBOARD[2]{ 6,9 };
    vector<vector<Point3f> > objpoints;
    vector<vector<Point2f> > imgpoints;
    vector<Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(Point3f(j, i, 0));
    }
    vector<String> images;
    string path = "./camera_cal/*.jpg";
    glob(path, images);

    Mat frame, gray;
    vector<Point2f> corner_pts;
    bool success;

    for (int i{ 0 }; i < images.size(); i++)
    {
        frame = imread(images[i]);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (success)
        {
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001);
            cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);
            drawChessboardCorners(frame, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
    }
    Mat R, T;
    calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
}

void perspective_warp(Mat& img, Mat& dst) {
    Mat final(720, 1280, CV_8UC3);
    Point2f trans[] = { Point2f(0,0), Point2f(1280,0), Point2f(1280,720), Point2f(0,720)};
    Point2f noktalar[] = { Point2f(500,500),Point2f(820,500),Point2f(1150,665),Point2f(160,665) };
    //sirasiyla sol ust, sag ust, sag alt, sol alt
    Mat M = getPerspectiveTransform(noktalar, trans);
    dst = getPerspectiveTransform(trans, noktalar);
    warpPerspective(img, img, M, img.size());
}

Mat abs_sobel_thresh(const Mat& gray, char orient = 'x', int sobel_kernel = 3, pair<int, int> thresh = make_pair(0, 255))
{
    Mat abs_sobel, binary_output;
    if (orient == 'x') {
        Sobel(gray, abs_sobel, CV_64F, 1, 0, sobel_kernel);
    }
    else if (orient == 'y') {
        Sobel(gray, abs_sobel, CV_64F, 0, 1, sobel_kernel);
    }
    convertScaleAbs(abs_sobel, abs_sobel);
    Mat scaled_sobel;
    normalize(abs_sobel, scaled_sobel, 0, 255, NORM_MINMAX, CV_8U);
    binary_output = Mat::zeros(scaled_sobel.size(), CV_8U);
    inRange(scaled_sobel, thresh.first, thresh.second, binary_output);

    return binary_output;
}

Mat mag_thresh(const Mat& gray, int sobel_kernel = 3, pair<int, int> mag_thresh = make_pair(0, 255))
{
    Mat sobelx, sobely, gradmag;

    Sobel(gray, sobelx, CV_64F, 1, 0, sobel_kernel);
    Sobel(gray, sobely, CV_64F, 0, 1, sobel_kernel);
    magnitude(sobelx, sobely, gradmag);
    double scale_factor = 255.0 / norm(gradmag, NORM_INF);
    gradmag.convertTo(gradmag, CV_8U, scale_factor);
    Mat binary_output = Mat::zeros(gradmag.size(), CV_8U);
    inRange(gradmag, mag_thresh.first, mag_thresh.second, binary_output);

    return binary_output;
}

Mat dir_threshold(const Mat& gray, int sobel_kernel = 3, pair<double, double> thresh = make_pair(0, CV_PI / 2))
{
    Mat sobelx, sobely, absgraddir;

    Sobel(gray, sobelx, CV_64F, 1, 0, sobel_kernel);
    Sobel(gray, sobely, CV_64F, 0, 1, sobel_kernel);

    Mat abs_sobelx = abs(sobelx);
    Mat abs_sobely = abs(sobely);

    Mat abs_sobelx_32f, abs_sobely_32f;
    abs_sobelx.convertTo(abs_sobelx_32f, CV_32F);
    abs_sobely.convertTo(abs_sobely_32f, CV_32F);

    Mat dir;
    divide(abs_sobely_32f, abs_sobelx_32f, dir);

    Mat pi = Mat::ones(dir.size(), CV_32F) * CV_PI / 2;
    Mat absgraddir_f = Mat::zeros(dir.size(), CV_32F);
    subtract(pi, dir, absgraddir_f);
    multiply(abs_sobelx_32f + abs_sobely_32f, absgraddir_f, absgraddir_f);
    absgraddir_f.convertTo(absgraddir, CV_8U);

    Mat binary_output = Mat::zeros(absgraddir.size(), CV_8U);
    inRange(absgraddir, thresh.first, thresh.second, binary_output);

    return binary_output;
}

Mat color_threshold_luv(Mat image, pair<int, int> l_thresh) {
    Mat l;
    cvtColor(image, l, COLOR_RGB2Luv);
    vector<Mat> luv_planes;
    split(l, luv_planes);
    Mat l_channel = luv_planes[0];

    Mat s_binary = Mat::zeros(l_channel.size(), CV_8UC1);
    inRange(l_channel, l_thresh.first, l_thresh.second, s_binary);

    return s_binary;
}

Mat color_threshold_lab(Mat image, pair<int, int> b_thresh, pair<int, int> l_thresh) {
    Mat lab;
    cvtColor(image, lab, COLOR_RGB2Lab);

    vector<Mat> lab_planes;
    split(lab, lab_planes);

    Mat b_channel = lab_planes[2];
    Mat l_channel = lab_planes[0];

    Mat l_binary = Mat::zeros(l_channel.size(), CV_8UC1);
    inRange(l_channel, l_thresh.first, l_thresh.second, l_binary);

    Mat b_binary = Mat::zeros(b_channel.size(), CV_8UC1);
    inRange(b_channel, b_thresh.first, b_thresh.second, b_binary);

    Mat combined_binary = Mat::zeros(b_channel.size(), CV_8UC1);
    bitwise_or(l_binary, b_binary, combined_binary);

    return combined_binary;
}

Mat get_mask(Mat frame) {
    Mat hls, gradx, grady, mag_binary, dir_binary, color_binary_luv, color_binary_lab;
    cvtColor(frame, hls, COLOR_BGR2HLS);
    Mat s_channel(hls.rows, hls.cols, CV_8UC1);
    int from_to[] = { 2, 0 };
    mixChannels(&hls, 1, &s_channel, 1, from_to, 1);

    gradx = abs_sobel_thresh(s_channel, 'x', 3, pair<int, int>(20, 100));
    grady = abs_sobel_thresh(s_channel, 'y', 3, pair<int, int>(20, 100));

    mag_binary = mag_thresh(s_channel, 3, pair<int, int>(20, 100));

    dir_binary = dir_threshold(s_channel, 3, pair<int, int>(0.7, 1.3));

    Mat g_combined = Mat::zeros(dir_binary.rows, dir_binary.cols, dir_binary.type());

    for (int i = 0; i < gradx.rows; i++) {
        for (int j = 0; j < gradx.cols; j++) {
            if (((gradx.at<uchar>(i, j) == 255) && (grady.at<uchar>(i, j) == 255)) ||
                ((mag_binary.at<uchar>(i, j) == 255) && (dir_binary.at<uchar>(i, j) == 255))) {
                 g_combined.at<uchar>(i, j) = 255;
            }
        }
    }
    morphologyEx(g_combined, g_combined, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)), Point(-1, -1), 1);
    
    color_binary_luv = color_threshold_luv(frame, pair<int, int>(170, 255));
    color_binary_lab = color_threshold_lab(frame, pair<int, int>(155, 255), pair<int, int>(170, 255));

    Mat combined_binary = Mat::zeros(dir_binary.size(), dir_binary.type());
    combined_binary.setTo(0);
    bitwise_or(color_binary_luv, color_binary_lab, combined_binary);

    Mat all_combined = Mat::zeros(frame.size(), CV_8UC1);
    all_combined.setTo(0);
    bitwise_or(g_combined, combined_binary, all_combined);

    return all_combined;
}

void polyfit(const vector<double>& t,
    const vector<double>& v,
    vector<double>& coeff,
    int order)
{
    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
    Eigen::VectorXd result;

    // check to make sure inputs are correct
    assert(t.size() == v.size());
    assert(t.size() >= order + 1);
    // Populate the matrix
    for (size_t i = 0; i < t.size(); ++i)
    {
        for (size_t j = 0; j < order + 1; ++j)
        {
            T(i, j) = pow(t.at(i), j);
        }
    }
    // Solve for linear least square fit
    result = T.householderQr().solve(V);
    coeff.resize(order + 1);

    int i = order;
    for (int k = 0; k < order + 1; k++)
    {
        coeff[i - k] = result[k];
    }

}

Mat find_lanes(Mat binary_warped, vector<double>& left_fitx, vector<double>& right_fitx, vector<double>& ploty, int nwindows = 9) {
    Mat bottom_half = binary_warped(Rect(0, binary_warped.rows / 2, binary_warped.cols, binary_warped.rows / 2));

    Mat out_img;
    cvtColor(binary_warped, out_img, COLOR_GRAY2BGR);
    out_img *= 255;

    int midpoint = binary_warped.cols / 2;
    Mat left_half = binary_warped(Rect(0, 0, midpoint, 720));
    Mat right_half = binary_warped(Rect(midpoint, 0, binary_warped.cols - midpoint, 720));
     
    Point left, rightt;
    int right;
    minMaxLoc(left_half, NULL, NULL, NULL, &left);
    minMaxLoc(right_half, NULL, NULL, NULL, &rightt);
    right = rightt.x + midpoint;

    int window_height = binary_warped.cols / nwindows;
    vector<Point> nonzero;
    
    findNonZero(binary_warped, nonzero);
    vector<int> nonzeroy(nonzero.size());
    vector<int> nonzerox(nonzero.size());

    for (int i = 0; i < nonzero.size(); i++)
    {
        nonzeroy[i] = nonzero[i].y;
        nonzerox[i] = nonzero[i].x;
    }

    int leftx_current = left.x;
    int rightx_current = right;

    int margin = 100;
    int minpix = 50;

    vector<int> left_lane_inds, right_lane_inds;
    for (int window = 0; window < nwindows; window++) {
        int win_y_low = binary_warped.cols - (window + 1) * window_height;
        int win_y_high = binary_warped.cols - window * window_height;

        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        rectangle(out_img, Point(win_xleft_low, win_y_low), Point(win_xleft_high, win_y_high), Scalar(0, 255, 0), 2);
        rectangle(out_img, Point(win_xright_low, win_y_low), Point(win_xright_high, win_y_high), Scalar(0, 255, 0), 2);

        for (int i = 0; i < nonzeroy.size(); i++) {
            if (nonzeroy[i] >= win_y_low && nonzeroy[i] < win_y_high) {
                if (nonzerox[i] >= win_xleft_low && nonzerox[i] < win_xleft_high) {
                    left_lane_inds.push_back(i);
                }
                if (nonzerox[i] >= win_xright_low && nonzerox[i] < win_xright_high) {
                    right_lane_inds.push_back(i);
                }
            }
        }
        
        if (left_lane_inds.size() > minpix) {
            int sum_left_x = 0;
            int count_left = 0;
            for (int i : left_lane_inds) {
                sum_left_x += nonzerox[i];
                count_left++;
            }
            leftx_current = sum_left_x / count_left;
        }
        if (right_lane_inds.size() > minpix) {
            int sum_right_x = 0;
            int count_right = 0;
            for (int i : right_lane_inds) {
                sum_right_x += nonzerox[i];
                count_right++;
            }
            rightx_current = sum_right_x / count_right;
        }
    }

    vector<int> concatenated_inds(left_lane_inds.size());
    vector<int> concatenated_inds_right(right_lane_inds.size());

    copy(left_lane_inds.begin(), left_lane_inds.end(), concatenated_inds.begin());
    copy(right_lane_inds.begin(), right_lane_inds.end(), concatenated_inds_right.begin());

    left_lane_inds = concatenated_inds;
    right_lane_inds = concatenated_inds_right;

    vector<double> leftx, lefty, rightx, righty;

    Vec3b red(0, 0, 255);
    Vec3b blue(255, 0, 0);

    for (int i = 0; i < left_lane_inds.size(); i++) {
        leftx.push_back(nonzerox[left_lane_inds[i]]);
        lefty.push_back(nonzeroy[left_lane_inds[i]]);
        int y = nonzeroy[left_lane_inds[i]];
        int x = nonzerox[left_lane_inds[i]];
        out_img.at<Vec3b>(y, x) = blue;

    }

    for (int i = 0; i < right_lane_inds.size(); i++) {
        rightx.push_back(nonzerox[right_lane_inds[i]]);
        righty.push_back(nonzeroy[right_lane_inds[i]]);
        int y = nonzeroy[right_lane_inds[i]];
        int x = nonzerox[right_lane_inds[i]];
        out_img.at<Vec3b>(y, x) = red;
        
    }

    vector<double> coeff_left;
    vector<double> coeff_right;

    polyfit(lefty, leftx, coeff_left, 2);
    polyfit(righty, rightx, coeff_right, 2);

    for (int i = 0; i < binary_warped.rows; i++) {
        ploty[i] = i;
    }

    for (int i = 0; i < ploty.size(); i++) {
        left_fitx[i] = coeff_left[0] * pow(ploty[i], 2) + coeff_left[1] * ploty[i] + coeff_left[2];
        right_fitx[i] = coeff_right[0] * pow(ploty[i], 2) + coeff_right[1] * ploty[i] + coeff_right[2];
    }
    return out_img;
}

Mat draw_lines(Mat image, Mat binary_warped, vector<double> left_fitx, vector<double> right_fitx, vector<double>& ploty, Mat Minv) {
    Mat warp_zero = Mat::zeros(binary_warped.size(), CV_8UC1);
    Mat color_warp;
    cvtColor(warp_zero, color_warp, COLOR_GRAY2BGR);

    vector<Point> pts_left;
    for (int i = 0; i < left_fitx.size(); i++) {
        Point pt;
        pt.x = (int)left_fitx[i];
        pt.y = (int)ploty[i];
        pts_left.push_back(pt);
    }

    vector<Point> pts_right;
    for (int i = 0; i < right_fitx.size(); i++) {
        Point pt;
        pt.x = (int)right_fitx[i];
        pt.y = (int)ploty[i];
        pts_right.push_back(pt);
    }
    reverse(pts_right.begin(), pts_right.end());

    vector<Point> pts;
    pts.insert(pts.end(), pts_left.begin(), pts_left.end());
    pts.insert(pts.end(), pts_right.begin(), pts_right.end());

    vector<vector<Point>> pts_poly;
    pts_poly.push_back(pts);

    Scalar color = Scalar(0, 255, 0);
    fillPoly(color_warp, pts_poly, color);

    Mat newwarp;
    warpPerspective(color_warp, newwarp, Minv, image.size());

    Mat result;
    addWeighted(image, 1, newwarp, 0.3, 0, result);

    return result;
}

int main() {
    Mat cameraMatrix, distCoeffs, undistFrame, result_g, Minv, lanes, final, mainFrame;
    cameraCalib(cameraMatrix, distCoeffs);
    
	VideoCapture video("car5.mp4");
	if (!video.isOpened()) {
		cout << "Video not found...";
		return -1;
	}
	while (true) {
		Mat frame, gray;
		bool hata = video.read(frame);
		if (!hata) {
			cout << "Data could not be read...";
			break;
		}
        undistort(frame, undistFrame, cameraMatrix, distCoeffs);
        undistFrame.copyTo(mainFrame);
        perspective_warp(undistFrame, Minv);
        result_g = get_mask(undistFrame);

        vector<double> left_fitx(result_g.rows), right_fitx(result_g.rows), ploty(result_g.rows);
        lanes = find_lanes(result_g, left_fitx, right_fitx, ploty);
        final = draw_lines(mainFrame, result_g, left_fitx, right_fitx, ploty, Minv);
        
        imshow("Find Lanes", lanes);
        imshow("Final Frame", final);

        if (waitKey(20) == 27){
            cout << "Esc button pressed...";
			break;
        }
	}

	return 0;
}