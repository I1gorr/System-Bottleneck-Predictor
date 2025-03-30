#include "ProcessInfo.h"
#include "HighCPUProcesses.h"  
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <thread>

using namespace std;
namespace fs = filesystem;


mutex dataMutex;
vector<ProcessInfo> highCPUProcesses;


string getProcessName(const string &pid) {
    ifstream cmdFile("/proc/" + pid + "/comm");
    string processName;
    if (cmdFile.is_open() && getline(cmdFile, processName)) {
        return processName;
    }
    return "Unknown";
}

// Function to get memory usage (RAM) from /proc/<PID>/status
string getProcessMemoryUsage(const string &pid) {
    ifstream statusFile("/proc/" + pid + "/status");
    string line;
    if (!statusFile.is_open()) return "N/A";

    while (getline(statusFile, line)) {
        if (line.find("VmRSS:") == 0) {  
            return line.substr(8);
        }
    }
    return "N/A";
}


long getProcessCPUTime(const string &pid) {
    ifstream statFile("/proc/" + pid + "/stat");
    string line;
    if (!statFile.is_open()) return -1;

    getline(statFile, line);
    stringstream ss(line);
    vector<string> tokens;
    string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    if (tokens.size() < 17) return -1;

    return stol(tokens[13]) + stol(tokens[14]);  // User + Kernel CPU time
}


double getCPUUsage(const string &pid) {
    long cpuTime1 = getProcessCPUTime(pid);  
    usleep(100000);  
    long cpuTime2 = getProcessCPUTime(pid);  

    if (cpuTime1 == -1 || cpuTime2 == -1 || cpuTime1 == cpuTime2) return 0.0;

    return ((cpuTime2 - cpuTime1) / (double) sysconf(_SC_CLK_TCK)) * 100;
}


void scanProcesses(const vector<string> &pids) {
    vector<ProcessInfo> localResults;
    
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);
    ostringstream dateStream;
    dateStream << put_time(localTime, "%Y-%m-%d");
    string formattedDate = dateStream.str();

    for (const string &pid : pids) {
        double cpuUsage = getCPUUsage(pid);
        if (cpuUsage > 1.0) {  
            localResults.push_back({
                pid,
                getProcessName(pid),
                getProcessMemoryUsage(pid),
                formattedDate,
                cpuUsage
            });
        }
    }

    lock_guard<mutex> lock(dataMutex);
    highCPUProcesses.insert(highCPUProcesses.end(), localResults.begin(), localResults.end());
}


void listHighCPUProcesses() {
    vector<string> allPids;

    for (const auto &entry : fs::directory_iterator("/proc")) {
        if (entry.is_directory()) {
            string pid = entry.path().filename();
            if (all_of(pid.begin(), pid.end(), ::isdigit)) {  
                allPids.push_back(pid);
            }
        }
    }

    size_t quarter = allPids.size() / 4;
    vector<string> part1(allPids.begin(), allPids.begin() + quarter);
    vector<string> part2(allPids.begin() + quarter, allPids.begin() + 2 * quarter);
    vector<string> part3(allPids.begin() + 2 * quarter, allPids.begin() + 3 * quarter);
    vector<string> part4(allPids.begin() + 3 * quarter, allPids.end());

    thread t1(scanProcesses, part1);
    thread t2(scanProcesses, part2);
    thread t3(scanProcesses, part3);
    thread t4(scanProcesses, part4);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    sort(highCPUProcesses.begin(), highCPUProcesses.end());

    cout << left << setw(20) << "Date"
         << setw(10) << "PID"
         << setw(25) << "Process Name"
         << setw(15) << "Memory Usage"
         << setw(10) << "CPU (%)" << endl;
    cout << "---------------------------------------------------------------------------\n";

    for (size_t i = 0; i < min(highCPUProcesses.size(), size_t(10)); i++) {
        const auto &p = highCPUProcesses[i];
        cout << left << setw(20)<< p.date
             << setw(10) << p.pid
             << setw(25) << p.name
             << setw(15) << p.memUsage
             << setw(10) << fixed << setprecision(2) << p.cpuUsage << "%" << endl;
    }
}


void exportHighCPUProcesses(const string &filename) {
    ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file " << filename << " for writing!" << endl;
        return;
    }

    outFile << "Date,PID,Process Name,Memory Usage,CPU (%)\n";


    for (const auto &p : highCPUProcesses) {
        outFile <<p.date<<","
                << p.pid << ","
                << p.name << ","
                << p.memUsage << ","
                << fixed << setprecision(2) << p.cpuUsage << "%\n";
    }

    outFile.close();
    cout << "High CPU processes exported to " << filename << endl;
}
