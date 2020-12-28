/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 50;  // TODO: Set the number of particles
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle temp_particle;
    temp_particle.id  = i;
    temp_particle.theta = theta;
    temp_particle.weight = 1.0;
    temp_particle.x += dist_x(gen);
    temp_particle.y += dist_y(gen);
    temp_particle.theta += dist_theta(gen);
    particles.push_back(temp_particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
                                  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    double part_x = particles[i].x;
    double part_y = particles[i].y;
    double part_theta = particles[i].theta;

    double p_x;
    double p_y;
    double p_theta;
    //Instead of a hard check of 0, adding a check for very low value of yaw_rate
    if (fabs(yaw_rate) < 0.0001) {
      p_x = part_x + velocity * cos(part_theta) * delta_t;
      p_y = part_y + velocity * sin(part_theta) * delta_t;
      p_theta = part_theta;
    } else {
      p_x = part_x + (velocity/yaw_rate) * (sin(part_theta + (yaw_rate * delta_t)) - sin(part_theta));
      p_y = part_y + (velocity/yaw_rate) * (cos(part_theta) - cos(part_theta + (yaw_rate * delta_t)));
      p_theta = part_theta + (yaw_rate * delta_t);
    }

    normal_distribution<double> dist_x(p_x, std_pos[0]);
    normal_distribution<double> dist_y(p_y, std_pos[1]);
    normal_distribution<double> dist_theta(p_theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {

    for (unsigned int i=0; i < observations.size(); i++) {
      LandmarkObs obs = observations[i];
      double minimum_distance = numeric_limits<double>::max();
      int map_id = -1;
      for (unsigned int k=0; k < predicted.size(); k++) {
        LandmarkObs pred = predicted[k];
        double current_distance = dist(obs.x, obs.y, pred.x, pred.y);
        if (current_distance < minimum_distance) {
          minimum_distance = current_distance;
          map_id = pred.id;
        }
      }
      observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

 for (int i = 0; i < num_particles; i++) {
   double px = particles[i].x;
   double py = particles[i].y;
   double ptheta = particles[i].theta;
   vector<LandmarkObs> predictions;
   for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
     float lmx = map_landmarks.landmark_list[j].x_f;
     float lmy = map_landmarks.landmark_list[j].y_f;
     int lmid = map_landmarks.landmark_list[j].id_i;
     if(dist(px, py, lmx, lmy) < sensor_range){
       predictions.push_back(LandmarkObs{ lmid, lmx, lmy });
     }
   }
   vector<LandmarkObs> tobs;
   for (unsigned int j = 0; j < observations.size(); j++) {
     double t_x = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
     double t_y = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
     tobs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
   }
   dataAssociation(predictions, tobs);
   particles[i].weight = 1.0;
   for (unsigned int j = 0; j < tobs.size(); j++) {
     double obx, oby, px, py;
     obx = tobs[j].x;
     oby = tobs[j].y;
     int associated_prediction = tobs[j].id;
     for (unsigned int k = 0; k < predictions.size(); k++) {
       if (predictions[k].id == associated_prediction) {
         px = predictions[k].x;
         py = predictions[k].y;
       }
     }
     double sx = std_landmark[0];
     double sy = std_landmark[1];
     double obs_w = ( 1/(2*M_PI*sx*sy)) * exp( -( pow(px-obx,2)/(2*pow(sx, 2)) + (pow(py-oby,2)/(2*pow(sy, 2))) ) );
     if(obs_w==0){
       obs_w=0.0001;
     }
     particles[i].weight *= obs_w;
   }
 }
}

void ParticleFilter::resample() {

   vector<Particle> new_particles;
   default_random_engine gen;
   vector<double> weights;
   for (int k = 0; k < num_particles; k++) {
     weights.push_back(particles[k].weight);
   }

   uniform_int_distribution<int> dist_index(0, num_particles - 1);
   int i = dist_index(gen);

   double max_weight = *max_element(weights.begin(), weights.end());
   double beta = 2*max_weight;

   uniform_real_distribution<double> dist_weight(0.0, max_weight);
   for (unsigned int j = 0; j < num_particles; j++) {

     beta = max_weight * 2;
     while (beta > weights[i]) {
       beta -= weights[i];
       i = (i + 1) % num_particles;
     }
     new_particles.push_back(particles[i]);
   }
   particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
