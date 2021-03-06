/* Pin layout
 *  Switch Pin ---> Clamp control
 *  Red ---> GND  
 *  White ---> 5V  
 *  Black ---> Signal 1 
 *  Green ---> Signal 2 
 *  
 *  Clamp switches armrest pin layout:
 *     up/down    open/close
 *P:  9       10      11     12   
 *    Raise 1 Raise 2 Open 1 Open 2
 *    R''     B''     R      B
 *  
 *  
 *  Limit switch ---> Arduino
 *  Clamp_movement & Clamp_grasp
 *  COM ---> 5V
 *  NC ---> R, GNDx
 *  
 *  FSR ---> Arduino 
 *  FSR_R ---> 5V
 *  FSR_L ---> R-GND, A0 
 *  
 *  Stretch sensor ---> Arduino
 *  SS_r ---> 5V 
 *  SS_l ---> 10k R -- GND, A1
 *  
 */

#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int16.h>


void switchCallback(const std_msgs::Bool&);
void clampmovementCallback(const std_msgs::Float32&);
void clampgraspCallback(const std_msgs::Float32&);

// Limit switch
/* Digital Read Pins
int limit_switch_up = 7;
int limit_switch_down = 6;
int limit_switch_open = 5;
int limit_switch_close = 4;
*/
/* Analog Read Pins */
// These pins are used because there is noise in the digital signal when all of the
// arduinos are plugged in and the motors are turned on. Using the analogRead() method
// allows for changing the threshold to a higher value for determining a HIGH or LOW
// signal.
int limit_switch_up = A3;
int limit_switch_down = A4;
int limit_switch_open = A5;
int limit_switch_close = A6;
int limit_switch_plate = A1;
int analog_threshold = 950; // if analogRead is above, switch is OFF, if below switch is ON

const int led = 13;
bool switch_status_up;
bool switch_status_down;
bool switch_status_open;
bool switch_status_close;
bool switch_status_plate;
int16_t switch_status_up_analog;
int16_t switch_status_down_analog;
int16_t switch_status_open_analog;
int16_t switch_status_close_analog;
int16_t switch_status_plate_analog;

// Force Sensitive Resistor
int fsrPin = A0;
int16_t fsrReading;
int force_threshold = 850;

// Stretch sensor
//int stretch_sensor_1_pin = A1;
int stretch_sensor_2_pin = A2;

// Clamp switch
const int RAISE_PIN_SIGNAL_1 = 9;
const int RAISE_PIN_SIGNAL_2 = 10;

const int OPEN_PIN_SIGNAL_1 = 11;
const int OPEN_PIN_SIGNAL_2 = 12;

const int PWM_MIN_1 = 120;
const int PWM_MIDDLE_1 = 190;
const int PWM_MAX_1 = 255;

const int PWM_MIN_2 = 80;
const int PWM_MIDDLE_2 = 127;
const int PWM_MAX_2 = 170;

bool clamp_switch;
float clamp_movement;
float clamp_grasp;

// ROS
ros::NodeHandle nh;

// Limit switch
std_msgs::Bool switch_up_msg;
std_msgs::Bool switch_down_msg;
std_msgs::Bool switch_open_msg;
std_msgs::Bool switch_close_msg;
std_msgs::Bool switch_plate_msg;
std_msgs::Int16 switch_analog_debug; // print out the analogRead value for debugging
ros::Publisher limit_switch_up_pub("switch_status_up", &switch_up_msg);
ros::Publisher limit_switch_down_pub("switch_status_down", &switch_down_msg);
ros::Publisher limit_switch_open_pub("switch_status_open", &switch_open_msg);
ros::Publisher limit_switch_close_pub("switch_status_close", &switch_close_msg);
ros::Publisher limit_switch_plate_pub("switch_status_plate", &switch_plate_msg);
ros::Publisher limit_switch_analog_pub("switch_status_analog", &switch_analog_debug);

// FSR
std_msgs::Int16 force_msg;
ros::Publisher force_pub("force", &force_msg);

// Stretch sensor
std_msgs::Float32 stretch_msg;
ros::Publisher stretch_sensor_pub("stretch_length", &stretch_msg);

// Clamp switch 
ros::Subscriber<std_msgs::Bool> clamp_switch_sub("clamp_control/clamp_switch", &switchCallback);
ros::Subscriber<std_msgs::Float32> clamp_movement_sub("clamp_control/clamp_movement", &clampmovementCallback);
ros::Subscriber<std_msgs::Float32> clamp_grasp_sub("clamp_control/clamp_grasp", &clampgraspCallback);
//
void setup() 
{
  nh.initNode();
  
  // Limit switch
  pinMode(limit_switch_up, INPUT);
  pinMode(led, OUTPUT);
  digitalWrite(led,HIGH);
  nh.advertise(limit_switch_up_pub);

  pinMode(limit_switch_down, INPUT);
  digitalWrite(led,HIGH);
  nh.advertise(limit_switch_down_pub);

  pinMode(limit_switch_open, INPUT);
  digitalWrite(led,HIGH);
  nh.advertise(limit_switch_open_pub);

  pinMode(limit_switch_close, INPUT);
  pinMode(led, OUTPUT);
  digitalWrite(led, HIGH);
  nh.advertise(limit_switch_close_pub);

  pinMode(limit_switch_plate, INPUT);
  nh.advertise(limit_switch_plate_pub);

  // For debugging the analogRead values
  nh.advertise(limit_switch_analog_pub);

  // FSR
  nh.advertise(force_pub);

  // Stretch sensor
  nh.advertise(stretch_sensor_pub);
  
  // Clamp
  nh.subscribe(clamp_switch_sub);
  nh.subscribe(clamp_movement_sub);
  nh.subscribe(clamp_grasp_sub);

  pinMode(RAISE_PIN_SIGNAL_1, OUTPUT);
  pinMode(RAISE_PIN_SIGNAL_2, OUTPUT);
  pinMode(OPEN_PIN_SIGNAL_1, OUTPUT);
  pinMode(OPEN_PIN_SIGNAL_2, OUTPUT);

  analogWrite(RAISE_PIN_SIGNAL_1, PWM_MIDDLE_1);
  analogWrite(RAISE_PIN_SIGNAL_2, PWM_MIDDLE_2);
  analogWrite(OPEN_PIN_SIGNAL_1, PWM_MIDDLE_1);
  analogWrite(OPEN_PIN_SIGNAL_2, PWM_MIDDLE_2);

  Serial.begin(19200);
}

void loop() 
{
  //=============//
  // Read Sensors
  //=============//
  // Limit switch
  // Up
  if (analogRead(limit_switch_up) < analog_threshold) {
    switch_status_up = true;
  }
  else {
    switch_status_up = false;
  }
  switch_up_msg.data = switch_status_up;
  limit_switch_up_pub.publish(&switch_up_msg);
  delay(10);

  // Down
  if (analogRead(limit_switch_down) < analog_threshold) {
    switch_status_down = true;
  }
  else {
    switch_status_down = false;
  }
  switch_down_msg.data = switch_status_down;
  limit_switch_down_pub.publish(&switch_down_msg);
  delay(10);

  // Open
  if (analogRead(limit_switch_open) < analog_threshold) {
    switch_status_open = true;
  }
  else {
    switch_status_open = false;
  }
  switch_open_msg.data = switch_status_open;
  limit_switch_open_pub.publish(&switch_open_msg);

  // Close
  if (analogRead(limit_switch_close) < analog_threshold) {
    switch_status_close = true;
  }
  else {
    switch_status_close = false;
  }
  switch_close_msg.data = switch_status_close;
  limit_switch_close_pub.publish(&switch_close_msg);

  // Plate
  if (analogRead(limit_switch_plate) < analog_threshold) {
    switch_status_plate = true;
  }
  else {
    switch_status_plate = false;
  }
  switch_plate_msg.data = switch_status_plate;
  limit_switch_plate_pub.publish(&switch_plate_msg);

  // DEBUG: print analog value
  switch_analog_debug.data = analogRead(limit_switch_close);
  limit_switch_analog_pub.publish(&switch_analog_debug);
  
  // FSR
  fsrReading = analogRead(fsrPin);
  force_msg.data = fsrReading;
  force_pub.publish(&force_msg);

  // Stretch sensor
  float stretch_value;

  // Currently the wires for stretch sensor 1 are being used for the 
  // plate switch (analog pin A1)
//  int value_1;
//  int v_1_in = 5;
//  float v_1_out = 0;
//  float r_1_1 = 10;
//  float r_1_2 = 0;
//  float val_1;
//  float buffer_1 = 0;
//  value_1 = analogRead(stretch_sensor_1_pin);
//  v_1_out = (5.0 / 1023.0) * value_1;
//  buffer_1 = (v_1_in / v_1_out) - 1;
//  r_1_2 = r_1_1 / buffer_1;
//  val_1 = 1000 / r_1_2;
  
  int value_2;
  int v_2_in = 5;
  float v_2_out = 0;
  float r_2_1 = 10;
  float r_2_2 = 0;
  float val_2;
  float buffer_2 = 0;
  value_2 = analogRead(stretch_sensor_2_pin);
  v_2_out = (5.0 / 1023.0) * value_2;
  buffer_2 = (v_2_in / v_2_out) - 1;
  r_2_2 = r_2_1 / buffer_2;
  val_2 = 1000 / r_2_2;

  //stretch_value = (val_1 + val_2) / 2;
  // Currently only using one stretch sensor, the other was replaced with a limit switch
  stretch_value = val_2;

//  // DEBUG: manually set reading that is within the tolerance for plate movement
//  // so that the plate does not have to be manually moved for the close command
//  // to run.
//  stretch_value = 20;
  
  stretch_msg.data = stretch_value;
  stretch_sensor_pub.publish( &stretch_msg );

  //=====================//
  // Clamp switch control
  //=====================//
  // Clamp movement up and down
  // negative command = upward movement
  if (clamp_movement <= 0)
  {
  // Check for the limit switch
    if (switch_status_up == false)
    {
      int pwm_signal_move_1 = map(100*clamp_movement, -100, 100, PWM_MIN_1, PWM_MAX_1);
      int pwm_signal_move_2 = map(100*clamp_movement, -100, 100, PWM_MAX_2, PWM_MIN_2);

      analogWrite(RAISE_PIN_SIGNAL_1, pwm_signal_move_1);
      analogWrite(RAISE_PIN_SIGNAL_2, pwm_signal_move_2);
    }
    else
    {
      analogWrite(RAISE_PIN_SIGNAL_1, PWM_MIDDLE_1);
      analogWrite(RAISE_PIN_SIGNAL_2, PWM_MIDDLE_2);
    }
  }
  
  else if (clamp_movement > 0)
  {
    if (switch_status_down == false)
    {
      int pwm_signal_move_1 = map(100*clamp_movement, -100, 100, PWM_MIN_1, PWM_MAX_1);
      int pwm_signal_move_2 = map(100*clamp_movement, -100, 100, PWM_MAX_2, PWM_MIN_2);

      analogWrite(RAISE_PIN_SIGNAL_1, pwm_signal_move_1);
      analogWrite(RAISE_PIN_SIGNAL_2, pwm_signal_move_2);
    }
    else
    {
      analogWrite(RAISE_PIN_SIGNAL_1, PWM_MIDDLE_1);
      analogWrite(RAISE_PIN_SIGNAL_2, PWM_MIDDLE_2);
    }
  }
  
  // Clamp open and close
  
  if (clamp_grasp <= 0) 
  {
    // Checking for the force sensor and close switch 
    if (switch_status_close == false && fsrReading < force_threshold)
    {
      int pwm_signal_grasp_1 = map(100*clamp_grasp, -100, 100, PWM_MIN_1, PWM_MAX_1);
      int pwm_signal_grasp_2 = map(100*clamp_grasp, -100, 100, PWM_MAX_2, PWM_MIN_2);

      analogWrite(OPEN_PIN_SIGNAL_1, pwm_signal_grasp_1);
      analogWrite(OPEN_PIN_SIGNAL_2, pwm_signal_grasp_2);
    }
    else
    {
      analogWrite(OPEN_PIN_SIGNAL_1, PWM_MIDDLE_1);
      analogWrite(OPEN_PIN_SIGNAL_2, PWM_MIDDLE_2);
    }
  }
  else if (clamp_grasp > 0)
  {
    // Check for limit switch
    if (switch_status_open == false)
    {
      int pwm_signal_grasp_1 = map(100*clamp_grasp, -100, 100, PWM_MIN_1, PWM_MAX_1);
      int pwm_signal_grasp_2 = map(100*clamp_grasp, -100, 100, PWM_MAX_2, PWM_MIN_2);
  
      analogWrite(OPEN_PIN_SIGNAL_1, pwm_signal_grasp_1);
      analogWrite(OPEN_PIN_SIGNAL_2, pwm_signal_grasp_2);
    } 
    else
    {
      analogWrite(OPEN_PIN_SIGNAL_1, PWM_MIDDLE_1);
      analogWrite(OPEN_PIN_SIGNAL_2, PWM_MIDDLE_2);
    }
  }
  
  nh.spinOnce();
  delay(100);
}

// Clamp switch
void switchCallback(const std_msgs::Bool& msg)
{
  clamp_switch = msg.data;
}

// Clamp movement
void clampmovementCallback(const std_msgs::Float32& msg)
{
  clamp_movement = msg.data;
}

// Clamp grasp
void clampgraspCallback(const std_msgs::Float32& msg)
{
  clamp_grasp = msg.data;
}
