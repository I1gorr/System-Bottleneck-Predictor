#ifndef PROCESSINFO_H  
#define PROCESSINFO_H

#include <string>
#include <vector>
#include <mutex>
#include <ctime>

using namespace std;

struct ProcessInfo {
    string time;        // Time of recording (HH:MM:SS)
    string name;        // Process name
    string memUsage;    // Memory usage
    string date;        // Date of recording (YYYY-MM-DD)
    double cpuUsage;    // CPU usage in percent

    // Operator overload to allow sorting by CPU usage (highest first)
    bool operator<(const ProcessInfo &other) const {
        return cpuUsage > other.cpuUsage;
    }
};

// Mutex to synchronize access to shared data
extern mutex dataMutex;
extern vector<ProcessInfo> highCPUProcesses;  

// Function declarations
string getProcessName(const string &pid);
string getProcessMemoryUsage(const string &pid);
long getProcessCPUTime(const string &pid);
double getCPUUsage(const string &pid);
void scanProcesses(const vector<string> &pids);
void listHighCPUProcesses();
void exportHighCPUProcesses(const string &filename);

#endif  
    