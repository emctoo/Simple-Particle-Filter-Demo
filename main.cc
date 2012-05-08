#include <iostream>
#include <limits>
#include <cmath>
#include <functional>
#include <numeric>
#include <opencv2/opencv.hpp>

void create_particles(cv::Mat &particles, int particle_number, cv::Size im_size)
{    
    particles = cv::Mat::zeros(particle_number, 4, CV_32F);

    cv::Mat x = particles.col(0), y = particles.col(1);
    cv::randu(x, cv::Scalar(0), cv::Scalar(im_size.width));
    cv::randu(y, cv::Scalar(0), cv::Scalar(im_size.height));
}

void update_particles(cv::Mat &particles, double pos_stddev, double vel_stddev)
{
    int particle_number = particles.rows;
    for(int r=0; r<particle_number; r++) {
        particles.at<float>(r, 0) += particles.at<float>(r, 2);
        particles.at<float>(r, 1) += particles.at<float>(r, 3);
    }

    cv::Mat pos = cv::Mat::zeros(particle_number, 2, CV_32F);
    cv::randn(pos, cv::Scalar(0), cv::Scalar(1));
    pos = pos_stddev * pos;
    cv::Mat _pos = particles(cv::Rect(0, 0, 2, particle_number));
    _pos = _pos + pos;

    cv::Mat vel = cv::Mat::zeros(particle_number, 2, CV_32F);
    cv::randn(vel, cv::Scalar(0), cv::Scalar(1));
    vel = vel_stddev * vel;
    cv::Mat _vel = particles(cv::Rect(2, 0, 2, particle_number));
    _vel = _vel + vel;
}

void loglikehood_to_likehood(double *log_likehood, int particle_number)
{
    double sum = 0;
    for(int i=0; i<particle_number; ++i) {
        log_likehood[i] = exp(log_likehood[i]);
        sum += log_likehood[i];
    }
   for(int i=0; i<particle_number; ++i) {
        log_likehood[i] /= sum;
        log_likehood[i] += log_likehood[i-1];
    }
}

void calc_likehood(cv::Mat particles, cv::Mat image, double *log_likehood, double rgb_stddev)
{
    int particle_number = particles.rows;
    std::vector<cv::Mat> bgrs;
    cv::split(image, bgrs);

    long double half_log_2_pi = -0.9189385332046727;        // -0.5*log(2*pi)    
    long double log_rgb_stddev = -log(rgb_stddev);          // -log(sigma)  

    for(int i=0; i<particle_number; i++) {
        int r = (int)particles.at<float>(i, 0);
        int c = (int)particles.at<float>(i, 1);
        log_likehood[i] = (-1)*std::numeric_limits<long double>::infinity();
        if(r>=0 && r<image.cols && c>=0 && c<image.rows) {
            cv::Vec3i bgr_color;
            bgr_color[0] = cv::Mat(bgrs[0]).at<uchar>(c, r);
            bgr_color[1] = cv::Mat(bgrs[1]).at<uchar>(c, r);
            bgr_color[2] = cv::Mat(bgrs[2]).at<uchar>(c, r);

            cv::Vec3i red_color(0, 0, 0);
            double color_distance = pow(cv::norm(bgr_color, red_color),2);
            double exp_part = color_distance/(2*rgb_stddev*rgb_stddev);
            log_likehood[i] = half_log_2_pi + log_rgb_stddev - exp_part;            
        }
    }
    loglikehood_to_likehood(log_likehood, particle_number);
}

template <typename _T>
int search(_T *base, _T key, int length)
{
    if(key < base[0]) return 0;

    int bottom = 0, top = length - 1;
    while (1) {
        if (key == base[bottom]) { return bottom+1; }
        if (key == base[top]) { return (top < length - 1 ? top + 1 : top); }
        if (top == bottom + 1 && key > base[bottom] && key < base[top]) {
            return top;
        }

        int idx = (top + bottom)/2;
        if(key < base[idx]) { top = idx; } else { bottom = idx; }
    }
}

template <typename _T>
void histc(_T *base, int *index, int length)
{
    cv::Mat samples = cv::Mat::zeros(1, length, CV_32F);    
    cv::RNG(cv::getTickCount()).fill(samples, cv::RNG::UNIFORM, 0, 1);

    for (int i=0; i<length; i++) {
        _T key = (_T) samples.at<float>(0, i);
        index[i] = search(base, key, length);
    }
}

void resample_particles(cv::Mat &particles, double *likehood)
{
    // _particles before
    cv::Mat _particles = particles.clone();

    int particle_number = particles.rows;
    int index[particle_number];
    histc<double>(likehood, index, particle_number);

    for(int ith_particle=0; ith_particle<particle_number; ++ith_particle) {                
        cv::Mat particle = particles.row(ith_particle);
        _particles.row(index[ith_particle]).copyTo(particle);
    }
}

void draw_particles(cv::Mat particles, cv::Mat &im) {
    for(int i=0; i<particles.rows; ++i) {
        int r = int(particles.at<float>(i, 0));
        int c = int(particles.at<float>(i, 1));
        if(r <0 || r>im.cols || c<0 || c>im.rows) { continue; }
        cv::circle(im, cv::Point(r, c), 2, cv::Scalar(255, 255, 0),-1,8);
    }
}

int main(int argc, char **argv) 
{
    // cv::VideoCapture source("/home/zeta/data/video_file.avi");
    cv::VideoCapture source(0);
    assert(source.isOpened());

    cv::Mat frame;
    source >> frame;

    cv::Mat particles;
    int width = frame.cols, height = frame.rows, particle_number = 2000;
    create_particles(particles, particle_number, cv::Size(width, height));

    double pos_stddev = 25, vel_stddev = 15, rgb_stddev = 50;

    assert(frame.rows*frame.cols == frame.total());
    double likehood[frame.total()];

    while (true) {
        source >> frame;
        update_particles(particles, pos_stddev, vel_stddev);
        calc_likehood(particles, frame, likehood, rgb_stddev);
        resample_particles(particles, likehood);
        draw_particles(particles, frame);
        cv::imshow("frame", frame);

        if(cv::waitKey(40) >= 0) { break; }
    }
    return 0;
}
