#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>


uint8_t broadcastAddress[] = {0x64, 0xB7, 0x08, 0x9C, 0x5D, 0xE0};  // ENPH459_ESP32_Comms MAC

typedef struct struct_message {
    int id;
    int leftSpeed;   // -255 to +255 (negative = reverse, positive = forward)
    int rightSpeed;  // -255 to +255 (negative = reverse, positive = forward)
} struct_message;

struct_message command;

// simple message for setting a GPIO pin: pin number and value (0 = LOW, 1 = HIGH)
typedef struct gpio_message {
  uint8_t pin;
  uint8_t value;
} gpio_message;

// heartbeat message for connectivity verification
typedef struct heartbeat_message {
  uint32_t counter;
  uint32_t timestamp;
} heartbeat_message;

gpio_message gpioCmd;

esp_now_peer_info_t peerInfo;

const int ledPin = 0;  
const int freq = 5000;
const int channel = 0;
const int resolution = 8;

static uint32_t heartbeat_counter = 0;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
           mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  
  Serial.print("\r\n[TRANSMITTER MAC: ");
  Serial.print(WiFi.macAddress());
  Serial.print("] Sent to [RECEIVER MAC: ");
  Serial.print(macStr);
  Serial.print("] - Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAILED");
}

// Called when data is received via ESP-NOW
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  Serial.print("\n[RECEIVER MAC: ");
  Serial.print(WiFi.macAddress());
  Serial.print("] Received packet from [TRANSMITTER MAC: ");
  Serial.println(macStr);

  if (len >= (int)sizeof(gpio_message)) {
    // interpret incoming data as gpio_message
    memcpy(&gpioCmd, incomingData, sizeof(gpio_message));
    Serial.print("GPIO cmd - pin: "); Serial.print(gpioCmd.pin);
    Serial.print(" value: "); Serial.println(gpioCmd.value);

    // basic validation: pin number in a reasonable range for ESP32 (0-39)
    if (gpioCmd.pin <= 39) {
      // set pin to OUTPUT before writing
      pinMode(gpioCmd.pin, OUTPUT);
      if (gpioCmd.value)
        digitalWrite(gpioCmd.pin, HIGH);
      else
        digitalWrite(gpioCmd.pin, LOW);
    } else {
      Serial.println("Received pin number out of range, ignoring");
    }
  } else {
    Serial.println("Received data is too small to be a gpio_message");
  }
}


void setup() {

  // iniialize Serial Monitor
  Serial.begin(115200);
  delay(1000);  // Give serial monitor time to start
 
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  Serial.println("\n========================================");
  Serial.println("ESP32 ESP-NOW Device Initialized");
  Serial.print("This Device's MAC Address: ");
  Serial.println(WiFi.macAddress());
  Serial.print("Target Receiver MAC Address: ");
  Serial.print(String(broadcastAddress[0], HEX)); Serial.print(":");
  Serial.print(String(broadcastAddress[1], HEX)); Serial.print(":");
  Serial.print(String(broadcastAddress[2], HEX)); Serial.print(":");
  Serial.print(String(broadcastAddress[3], HEX)); Serial.print(":");
  Serial.print(String(broadcastAddress[4], HEX)); Serial.print(":");
  Serial.println(String(broadcastAddress[5], HEX));
  Serial.println("========================================\n");

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register send callback
  esp_now_register_send_cb(OnDataSent);
  
  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Peer registered successfully!");
  Serial.println("\n========== SPEED INPUT INSTRUCTIONS ===========");
  Serial.println("Enter motor speed values (-255 to +255):");
  Serial.println("  > Single value for both motors: 100");
  Serial.println("  > Separate values: 100 -75  (left=100, right=-75)");
  Serial.println("  > Negative = Reverse, Positive = Forward");
  Serial.println("  > Range: -255 to +255 (auto clamped)");
  Serial.println("==============================================\n");
}

void loop() {
  // Send heartbeat every 1 second
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat > 1000) {
    heartbeat_message hb;
    hb.counter = heartbeat_counter++;
    hb.timestamp = millis();
    
    Serial.print("[HEARTBEAT] Sending pulse #");
    Serial.print(hb.counter);
    Serial.println(" ms");
    
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&hb, sizeof(hb));
    if (result != ESP_OK) {
      Serial.println("Failed to send heartbeat");
    }
    lastHeartbeat = millis();
  }
  
  // Handle serial input for speed commands
  static int leftSpeed = 0;
  static int rightSpeed = 0;
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() > 0) {
      // Parse input format: "left right" (e.g., "-128 75" means left=-128, right=75)
      int spaceIndex = input.indexOf(' ');
      
      if (spaceIndex > 0) {
        // Two values provided
        int left = input.substring(0, spaceIndex).toInt();
        int right = input.substring(spaceIndex + 1).toInt();
        
        // Validate and clamp
        leftSpeed = constrain(left, -255, 255);
        rightSpeed = constrain(right, -255, 255);
        
        Serial.print("[INPUT] Set Left: ");
        Serial.print(leftSpeed);
        Serial.print(" Right: ");
        Serial.println(rightSpeed);
      } else {
        // Single value provided - apply to both
        int speed = input.toInt();
        leftSpeed = constrain(speed, -255, 255);
        rightSpeed = constrain(speed, -255, 255);
        
        Serial.print("[INPUT] Set Both to: ");
        Serial.println(leftSpeed);
      }
      
      Serial.println("  Format: 'left right' or just 'value' for both (-255 to +255)");
    }
  }
  
  // Send thrust commands at regular interval (100ms)
  static unsigned long lastThrust = 0;
  if (millis() - lastThrust > 100) {
    struct_message msg;
    msg.id = 1;
    msg.leftSpeed = leftSpeed;
    msg.rightSpeed = rightSpeed;
    
    Serial.print("[SPEED CMD] Left: ");
    Serial.print(msg.leftSpeed);
    Serial.print(" Right: ");
    Serial.println(msg.rightSpeed);
    
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&msg, sizeof(msg));
    if (result != ESP_OK) {
      Serial.println("Failed to send speed command");
    }
    lastThrust = millis();
  }
  
  delay(50);
}