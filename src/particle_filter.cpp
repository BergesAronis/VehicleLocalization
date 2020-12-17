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
  num_particles = 100;  // TODO: Set the number of particles
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(num_particles);
  weights.resize(num_particles);
  for (int i = 0; i < num_particles; i++) {
    Particle temp_particle;
    temp_particle.id  = i;
    temp_particle.x = dist_x(gen);
    temp_particle.y = dist_y(gen);
    temp_particle.theta = dist_theta(gen);
    temp_particle.weight = 1.0;
    particles.push_back(temp_particle);
    weights.push_back(temp_particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
   default_random_engine gen;
   normal_distribution<double> dist_x(0, std_pos[0]);
   normal_distribution<double> dist_y(0, std_pos[1]);
   normal_distribution<double> dist_theta(0, std_pos[2]);

   for (int i = 0; i < num_particles; i++) {

     if (fabs(yaw_rate) < 0.00001) {
       particles[i].x += delta_t * velocity * cos(particles[i].theta);
       particles[i].y += delta_t * velocity * sin(particles[i].theta);
     } else {
       particles[i].x += (sin(particles[i].theta + delta_t*yaw_rate) - sin(particles[i].theta)) * (velocity/yaw_rate);
       particles[i].y += (cos(particles[i].theta) - cos(particles[i].theta + delta_t*yaw_rate)) * (velocity/yaw_rate);
       particles[i].theta += yaw_rate * delta_t;
     }

     particles[i].x += dist_x(gen);
     particles[i].y += dist_y(gen);
     particles[i].theta += dist_theta(gen);
   }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {

    for (unsigned int i=0; i < observations.size(); i++) {
      double minimum_distance = numeric_limits<double>::max();
      int map_id = -1;
      for (unsigned int k=0; k < predicted.size(); k++) {
        double current_distance = pow((observations[i].x - predicted[k].x), 2) + pow((observations[i].y - predicted[k].y), 2);

        if (current_distance < minimum_distance) {
          minimum_distance = current_distance;
          map_id = predicted[k].id;
        }
      }
      observations[i].id = map_id;
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

   vector<LandmarkObs> predictions;
   for (int i = 0; i < num_particles; i++) {


     for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {

       if (fabs(map_landmarks.landmark_list[k].x_f - particles[i].x) <= sensor_range &&
           fabs(map_landmarks.landmark_list[k].y_f - particles[i].y) <= sensor_range) {
         predictions.push_back(LandmarkObs{ map_landmarks.landmark_list[k].id_i,
                                            map_landmarks.landmark_list[k].x_f,
                                            map_landmarks.landmark_list[k].y_f});
       }

     }

     vector<LandmarkObs> transformed_observations;
     for (unsigned int k = 0; k < observations.size(); k++) {
       LandmarkObs transformed_observation;
       transformed_observation.id = k;
       transformed_observation.x = cos(particles[i].theta) * observations[k].x - sin(particles[i].theta)*observations[k].y + particles[i].x;
       transformed_observation.y = sin(particles[i].theta) * observations[k].x + cos(particles[i].theta)*observations[k].y + particles[i].y;
       transformed_observations.push_back(transformed_observation);
     }

     dataAssociation(predictions, transformed_observations);

     particles[i].weight = 1.0;
     double px, py;
	 double weight = 1.0;
     for (unsigned int k = 0; k < transformed_observations.size(); k++) {
       for (unsigned int j = 0; j < predictions.size(); j++) {
         if (predictions[j].id == transformed_observations[k].id) {
           px = predictions[j].x;
           py = predictions[j].y;
          }
       }
       weight *= (1.0/2.0 * M_PI * std_landmark[0] * std_landmark[1])  * exp(-1.0 * ((pow((transformed_observations[k].x - px), 2)/(2.0 * pow(std_landmark[0], 2))) + (pow((transformed_observations[k].y - py), 2)/(2.0 * pow(std_landmark[1], 2)))));

     }
     particles[i].weight = weight;
     weights[i] = weight;
   }
}

void ParticleFilter::resample() {

   vector<Particle> new_particles;
   default_random_engine gen;

    uniform_int_distribution<int> dist_index(0, num_particles - 1);
   int i = dist_index(gen);

   double max_weight = numeric_limits<double>::min();
   double beta = 0.0;

   uniform_real_distribution<double> dist_weight(0.0, max_weight);
   for (unsigned int j = 0; j < particles.size(); j++) {

     beta += dist_weight(gen) * 2.0;
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
