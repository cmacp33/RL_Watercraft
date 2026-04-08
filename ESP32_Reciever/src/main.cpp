#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>


uint8_t broadcastAddress[] = {0x64, 0xB7, 0x08, 0x9D, 0x46, 0xA4};  // ENPH459_ESP32_Test2 MAC

typedef struct struct_message {
    int id;
    int leftSpeed;   // -255 to +255 (negative = reverse, positive = forward)
    int rightSpeed;  // -255 to +255 (negative = reverse, positive = forward)
} struct_message;

typedef struct gpio_message {
  uint8_t pin;
  uint8_t value;
} gpio_message;

// heartbeat message for connectivity verification
typedef struct heartbeat_message {
  uint32_t counter;
  uint32_t timestamp;
} heartbeat_message;

struct_message command;

esp_now_peer_info_t peerInfo;

const int ledPin = 0;  
const int LEFT_MOTOR_PIN = 2;
const int RIGHT_MOTOR_PIN = 5;
const int ESC_FREQ = 50;         // 50 Hz for ESC
const int ESC_RESOLUTION = 16;   // 16-bit PWM for fine control
const int LEFT_CHANNEL = 0;
const int RIGHT_CHANNEL = 1;

// ESC PWM signal timing (in microseconds for 50Hz)
const int ESC_MIN_PULSE = 1000;   // 1ms = full reverse
const int ESC_MID_PULSE = 1500;   // 1.5ms = neutral/stop
const int ESC_MAX_PULSE = 2000;   // 2ms = full forward

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nLast Packet Send Status:\t");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

// Convert speed value (-255 to +255) to ESC PWM pulse width (1000-2000 microseconds)
void setMotorSpeed(int channel, int speed) {
  // Constrain speed to valid range
  speed = constrain(speed, -255, 255);
  
  // Map speed to pulse width
  // Negative speed = reverse (1000-1500 us)
  // Positive speed = forward (1500-2000 us)
  int pulseWidth;
  if (speed < 0) {
    // Reverse: -255 to 0 maps to 1000 to 1500 us
    pulseWidth = ESC_MID_PULSE + (speed * (ESC_MID_PULSE - ESC_MIN_PULSE)) / 255;
  } else {
    // Forward: 0 to +255 maps to 1500 to 2000 us
    pulseWidth = ESC_MID_PULSE + (speed * (ESC_MAX_PULSE - ESC_MID_PULSE)) / 255;
  }
  
  // Convert microseconds to duty cycle for 16-bit resolution at 50Hz
  // 50Hz = 20000us period, 16-bit = 65536 steps
  // dutyCycle = (pulseWidth / 20000) * 65536
  int dutyCycle = (pulseWidth * 65536) / 20000;
  
  ledcWrite(channel, dutyCycle);
}

// Callback when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  
  // Check if it's a heartbeat message
  if (len == (int)sizeof(heartbeat_message)) {
    heartbeat_message hb;
    memcpy(&hb, incomingData, sizeof(heartbeat_message));
    
    Serial.print("[✓ HEARTBEAT] Pulse #");
    Serial.print(hb.counter);
    Serial.println(" - PEER IS ALIVE!");
    return;
  }
  
  // Check if it's a thrust/command message
  if (len == (int)sizeof(struct_message)) {
    struct_message cmd;
    memcpy(&cmd, incomingData, sizeof(struct_message));
    
    Serial.print("\n[SPEED CMD RECEIVED] Left: ");
    Serial.print(cmd.leftSpeed);
    Serial.print(" Right: ");
    Serial.println(cmd.rightSpeed);
    
    // Set motor speeds using proper ESC control
    setMotorSpeed(LEFT_CHANNEL, cmd.leftSpeed);
    setMotorSpeed(RIGHT_CHANNEL, cmd.rightSpeed);
    
    Serial.println("Motors updated!");
    return;
  }
  
  Serial.print("\n[WARNING] Received unknown message from ");
  Serial.print(macStr);
  Serial.print(" - size: ");
  Serial.println(len);
}


void setup() {

  // Initialize Serial Monitor
  Serial.begin(115200);
 
  delay(1000);
  Serial.println("\n\nESP-NOW Receiver Starting...");
  
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);
  
  Serial.print("This ESP-NOW MAC Address: ");
  Serial.println(WiFi.macAddress());
  
  // Setup PWM for ESC motors
  Serial.println("\nSetup PWM for ESC motors:");
  Serial.print("  Left Motor (Pin ");
  Serial.print(LEFT_MOTOR_PIN);
  Serial.print(") - Channel ");
  Serial.print(LEFT_CHANNEL);
  Serial.println(", 50 Hz, 16-bit");
  Serial.print("  Right Motor (Pin ");
  Serial.print(RIGHT_MOTOR_PIN);
  Serial.print(") - Channel ");
  Serial.print(RIGHT_CHANNEL);
  Serial.println(", 50 Hz, 16-bit");
  
  ledcSetup(LEFT_CHANNEL, ESC_FREQ, ESC_RESOLUTION);
  ledcSetup(RIGHT_CHANNEL, ESC_FREQ, ESC_RESOLUTION);
  ledcAttachPin(LEFT_MOTOR_PIN, LEFT_CHANNEL);
  ledcAttachPin(RIGHT_MOTOR_PIN, RIGHT_CHANNEL);
  
  // Start with both motors at neutral (0 speed)
  setMotorSpeed(LEFT_CHANNEL, 0);
  setMotorSpeed(RIGHT_CHANNEL, 0);
  Serial.println("PWM motors initialized!\n");

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register receive callback
  esp_now_register_recv_cb(OnDataRecv);
  
  // Register peer (optional, but good practice)
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Peer registered successfully!");
  Serial.println("Listening for thrust commands...");
}

void loop() {
  // Monitor heap memory and re-register peer periodically
  static unsigned long lastHeapCheck = 0;
  static unsigned long lastPeerRefresh = 0;
  
  unsigned long now = millis();
  
  // Print heap memory every 10 seconds (diagnostic)
  if (now - lastHeapCheck > 10000) {
    uint32_t freeHeap = ESP.getFreeHeap();
    Serial.print("[HEALTH] Free Heap: ");
    Serial.print(freeHeap);
    Serial.println(" bytes");
    
    if (freeHeap < 50000) {
      Serial.println("[WARNING] Low memory! Consider restarting.");
    }
    lastHeapCheck = now;
  }
  
  // Re-register peer every 30 seconds to keep connection fresh
  if (now - lastPeerRefresh > 30000) {
    Serial.println("[PEER REFRESH] Re-registering peer...");
    
    // Remove old peer if it exists
    esp_now_del_peer(broadcastAddress);
    delay(100);
    
    // Re-add peer
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;
    
    if (esp_now_add_peer(&peerInfo) == ESP_OK) {
      Serial.println("[PEER REFRESH] Peer re-registered successfully");
    } else {
      Serial.println("[ERROR] Failed to re-register peer!");
    }
    lastPeerRefresh = now;
  }
  
  delay(100);
}





/*
The non-wifi sending commands to control the ESCs
*/

// #include <Arduino.h>
// #include <esp_now.h>
// #include <WiFi.h>
// uint8_t broadcastAddress[] = {0x24, 0x6F, 0x28, 0xAA, 0xBB, 0xCC};
// typedef struct struct_message {
//     int id;
//     float leftThrust;
//     float rightThrust;
// } struct_message;
// struct_message command;
// esp_now_peer_info_t peerInfo;
// const int escPinLeft = 0;   // GPIO0
// const int escPinRight = 37; // GPIO37
// const int freq = 50;        // 50Hz for servo/ESC control
// const int channelLeft = 0;
// const int channelRight = 1;
// const int resolution = 16;  // 16-bit resolution for microseconds
// void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
//   Serial.print("\r\nLast Packet Send Status:\t");
//   Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
// }
// void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
//   memcpy(&command, incomingData, sizeof(command));
//   Serial.print("Left Thrust: ");
//   Serial.print(command.leftThrust);
//   Serial.print(" | Right Thrust: ");
//   Serial.println(command.rightThrust);
// }
// void handleSerialInput() {
//   if (Serial.available()) {
//     String input = Serial.readStringUntil('\n');
//     input.trim();
    
//     if (input.length() == 0) return;
    
//     char cmd = input.charAt(0);
//     String valueStr = input.substring(1);
//     valueStr.trim();
    
//     if (cmd == 'f' || cmd == 'F') {
//       float percentage = valueStr.toFloat();
//       command.leftThrust = percentage / 100.0;
//       command.rightThrust = percentage / 100.0;
//       Serial.print("Forward: ");
//       Serial.println(command.leftThrust);
//     } 
//     else if (cmd == 'r' || cmd == 'R') {
//       float percentage = valueStr.toFloat();
//       command.leftThrust = -(percentage / 100.0);
//       command.rightThrust = -(percentage / 100.0);
//       Serial.print("Reverse: ");
//       Serial.println(command.leftThrust);
//     }
//     else if (cmd == 'n' || cmd == 'N') {
//       command.leftThrust = 0;
//       command.rightThrust = 0;
//       Serial.println("Neutral");
//     }
//     else {
//       Serial.println("Commands: f [0-100], r [0-100], n");
//     }
//   }
// }
// void setup() {
//   Serial.begin(115200);
//   delay(1000);
//   Serial.println("\n\nESC Test Starting...");
 
//   WiFi.mode(WIFI_STA);
//   if (esp_now_init() != ESP_OK) {
//     Serial.println("Error initializing ESP-NOW");
//     return;
//   }
//   esp_now_register_recv_cb(OnDataRecv);
//   memcpy(peerInfo.peer_addr, broadcastAddress, 6);
//   peerInfo.channel = 0;
//   peerInfo.encrypt = false;
//   if (esp_now_add_peer(&peerInfo) != ESP_OK){
//     Serial.println("Failed to add peer");
//     return;
//   }

//   // Setup PWM for both ESC channels
//   ledcSetup(channelLeft, freq, resolution);
//   ledcAttachPin(escPinLeft, channelLeft);

//   ledcSetup(channelRight, freq, resolution);
//   ledcAttachPin(escPinRight, channelRight);
  
//   // Start both at neutral (1500 microseconds)
//   uint16_t neutralDuty = (1500 * 65535) / 20000;  // ≈ 4915
//   ledcWrite(channelLeft, neutralDuty);
//   ledcWrite(channelRight, neutralDuty);
//   delay(2000);
// }
// void loop() {
//   handleSerialInput();
  
//   // Left thrust: convert -1.0 to 1.0 → 1000 to 2000µs
//   float leftMicroseconds = 1500.0 + (command.leftThrust * 500.0);
//   leftMicroseconds = constrain(leftMicroseconds, 1000.0, 2000.0);
//   uint16_t leftDuty = (uint16_t)((leftMicroseconds * 65535) / 20000);

//   // Right thrust: convert -1.0 to 1.0 → 1000 to 2000µs
//   float rightMicroseconds = 1500.0 + (command.rightThrust * 500.0);
//   rightMicroseconds = constrain(rightMicroseconds, 1000.0, 2000.0);
//   uint16_t rightDuty = (uint16_t)((rightMicroseconds * 65535) / 20000);
  
//   Serial.print("Left Thrust: ");
//   Serial.print(command.leftThrust);
//   Serial.print(" | µs: ");
//   Serial.print(leftMicroseconds);
//   Serial.print(" | Duty: ");
//   Serial.print(leftDuty);
//   Serial.print("  ||  Right Thrust: ");
//   Serial.print(command.rightThrust);
//   Serial.print(" | µs: ");
//   Serial.print(rightMicroseconds);
//   Serial.print(" | Duty: ");
//   Serial.println(rightDuty);
  
//   ledcWrite(channelLeft, leftDuty);
//   ledcWrite(channelRight, rightDuty);
//   delay(200);
// }

// // put function definitions here:
// int myFunction(int x, int y) {
//   return x + y;
// }