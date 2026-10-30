#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>


uint8_t broadcastAddress[] = {0x44, 0x1B, 0xF6, 0x81, 0x7D, 0x4C};  // ESP32-S3 receiver (boat) MAC

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

// ---------- Serial dashboard state ----------
const int SPEED_STEP = 25;                 // change per hotkey press
const unsigned long UI_REFRESH_MS = 200;   // dashboard redraw interval
const unsigned long LINK_TIMEOUT_MS = 1500;

static int leftSpeed = 0;
static int rightSpeed = 0;
static String inputBuf;                    // characters typed so far (shown on screen)
static String statusMsg = "Ready";
static bool uiDirty = true;
static int escState = 0;                   // parser state for arrow-key escape sequences

// updated from the ESP-NOW (WiFi task) callback
volatile uint32_t sendOk = 0;
volatile uint32_t sendFail = 0;
volatile unsigned long lastAckMs = 0;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Count results instead of printing, so the dashboard isn't flooded
  if (status == ESP_NOW_SEND_SUCCESS) {
    sendOk++;
    lastAckMs = millis();
  } else {
    sendFail++;
  }
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

  // Clear screen and hide cursor; the dashboard redraws in place from loop()
  Serial.print("\x1b[2J\x1b[?25l");

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    statusMsg = "ERROR: ESP-NOW init failed";
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
    statusMsg = "ERROR: failed to add peer";
    return;
  }

  statusMsg = "Peer registered - ready";
}

void setSpeeds(int left, int right) {
  leftSpeed = constrain(left, -255, 255);
  rightSpeed = constrain(right, -255, 255);
  statusMsg = "Set L=" + String(leftSpeed) + " R=" + String(rightSpeed);
  uiDirty = true;
}

// Parse a typed line: "value" for both motors, or "left right"
void applyInputLine() {
  int left, right;
  int n = sscanf(inputBuf.c_str(), "%d %d", &left, &right);
  if (n == 2) {
    setSpeeds(left, right);
  } else if (n == 1) {
    setSpeeds(left, left);
  } else {
    statusMsg = "Invalid input: \"" + inputBuf + "\"";
  }
  inputBuf = "";
}

void handleKey(char c) {
  // Arrow keys arrive as ESC [ A/B/C/D
  if (escState == 1) { escState = (c == '[') ? 2 : 0; return; }
  if (escState == 2) {
    escState = 0;
    switch (c) {
      case 'A': setSpeeds(leftSpeed + SPEED_STEP, rightSpeed + SPEED_STEP); break;  // up
      case 'B': setSpeeds(leftSpeed - SPEED_STEP, rightSpeed - SPEED_STEP); break;  // down
      case 'D': setSpeeds(leftSpeed - SPEED_STEP, rightSpeed + SPEED_STEP); break;  // left
      case 'C': setSpeeds(leftSpeed + SPEED_STEP, rightSpeed - SPEED_STEP); break;  // right
    }
    return;
  }

  switch (c) {
    case 0x1B: escState = 1; return;
    case '\r':
    case '\n':
      if (inputBuf.length() > 0) applyInputLine();
      break;
    case '\b':
    case 0x7F:
      if (inputBuf.length() > 0) inputBuf.remove(inputBuf.length() - 1);
      break;
    case 'w': case 'W': setSpeeds(leftSpeed + SPEED_STEP, rightSpeed + SPEED_STEP); break;
    case 's': case 'S': setSpeeds(leftSpeed - SPEED_STEP, rightSpeed - SPEED_STEP); break;
    case 'a': case 'A': setSpeeds(leftSpeed - SPEED_STEP, rightSpeed + SPEED_STEP); break;
    case 'd': case 'D': setSpeeds(leftSpeed + SPEED_STEP, rightSpeed - SPEED_STEP); break;
    case 'x': case 'X':
      inputBuf = "";
      setSpeeds(0, 0);
      statusMsg = "STOP";
      break;
    default:
      if ((isDigit(c) || c == '-' || c == '+' || c == ' ') && inputBuf.length() < 16) {
        inputBuf += c;
      }
      break;
  }
  uiDirty = true;
}

// Draws e.g. "-------###|#######---" (reverse fills left of centre, forward fills right)
void printSpeedBar(int v) {
  const int half = 10;
  int n = (abs(v) * half + 127) / 255;
  char bar[2 * half + 2];
  for (int i = 0; i < half; i++) bar[i] = (v < 0 && i >= half - n) ? '#' : '-';
  bar[half] = '|';
  for (int i = 0; i < half; i++) bar[half + 1 + i] = (v > 0 && i < n) ? '#' : '-';
  bar[2 * half + 1] = '\0';
  Serial.print(bar);
}

void drawUI() {
  const char *EOL = "\x1b[K\r\n";  // clear rest of line, then newline

  Serial.print("\x1b[H");  // cursor to top-left
  Serial.printf("=== RL Watercraft Transmitter ===%s", EOL);
  Serial.printf("TX %s  ->  RX %02X:%02X:%02X:%02X:%02X:%02X%s", WiFi.macAddress().c_str(),
                broadcastAddress[0], broadcastAddress[1], broadcastAddress[2],
                broadcastAddress[3], broadcastAddress[4], broadcastAddress[5], EOL);

  unsigned long lastAck = lastAckMs;
  Serial.print("Link: ");
  if (lastAck == 0) {
    Serial.print("\x1b[33mWAITING\x1b[0m");
  } else if (millis() - lastAck < LINK_TIMEOUT_MS) {
    Serial.print("\x1b[32mOK\x1b[0m     ");
  } else {
    Serial.printf("\x1b[31mLOST\x1b[0m (%lus)", (millis() - lastAck) / 1000);
  }
  Serial.printf("   sent ok: %lu  failed: %lu%s", (unsigned long)sendOk, (unsigned long)sendFail, EOL);
  Serial.print(EOL);

  Serial.printf("LEFT  %+4d  ", leftSpeed);
  printSpeedBar(leftSpeed);
  Serial.print(EOL);
  Serial.printf("RIGHT %+4d  ", rightSpeed);
  printSpeedBar(rightSpeed);
  Serial.print(EOL);
  Serial.print(EOL);

  Serial.printf("Status: %s%s", statusMsg.c_str(), EOL);
  Serial.print(EOL);
  Serial.printf("Type \"value\" (both) or \"left right\" + Enter   (-255..255)%s", EOL);
  Serial.printf("W/S or Up/Down: both +/-%d   A/D or Left/Right: turn   X: STOP%s", SPEED_STEP, EOL);
  Serial.print(EOL);
  Serial.printf("> %s_%s", inputBuf.c_str(), EOL);
  Serial.print("\x1b[J");  // clear anything below
}

void loop() {
  // Send heartbeat every 1 second
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat > 1000) {
    heartbeat_message hb;
    hb.counter = heartbeat_counter++;
    hb.timestamp = millis();
    esp_now_send(broadcastAddress, (uint8_t *)&hb, sizeof(hb));
    lastHeartbeat = millis();
  }

  // Handle keyboard input one character at a time
  while (Serial.available()) {
    handleKey((char)Serial.read());
  }

  // Send thrust commands at regular interval (100ms)
  static unsigned long lastThrust = 0;
  if (millis() - lastThrust > 100) {
    struct_message msg;
    msg.id = 1;
    msg.leftSpeed = leftSpeed;
    msg.rightSpeed = rightSpeed;
    esp_now_send(broadcastAddress, (uint8_t *)&msg, sizeof(msg));
    lastThrust = millis();
  }

  // Redraw on a timer (for link stats) or immediately after a keypress
  static unsigned long lastDraw = 0;
  if (uiDirty || millis() - lastDraw > UI_REFRESH_MS) {
    drawUI();
    uiDirty = false;
    lastDraw = millis();
  }

  delay(5);
}
