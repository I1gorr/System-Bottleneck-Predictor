#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <vector>
#include <cstdlib>  
#include <chrono>
#include <thread>
#include <future>   
#include "ProcessInfo.h"
using namespace std;



// Function to get and print process details from processes.csv
vector<ProcessInfo> getProcessDetails() {
    ifstream file("processes.csv");
    if (!file.is_open()) {
        cerr << "Error: Unable to open processes.csv" << endl;
        return {};
    }

    string line;
    getline(file, line); // Skip header

    vector<ProcessInfo> processes;
    double totalCPUUsage = 0.0;

    cout << "\n----- Process Details from processes.csv -----\n";
    cout << "Date\t\tPID\tProcess Name\tMemory Usage\tCPU Usage\n";

    while (getline(file, line)) {
        stringstream ss(line);
        ProcessInfo process;

        getline(ss, process.date, ',');
        getline(ss, process.pid, ',');
        getline(ss, process.name, ',');
        getline(ss, process.memUsage, ',');
        ss >> process.cpuUsage;

        // Print process details
        cout << process.date << "\t" << process.pid << "\t" << process.name << "\t" 
             << process.memUsage << "\t" << process.cpuUsage << "%" << endl;

        totalCPUUsage += process.cpuUsage;
        processes.push_back(process);
    }

    file.close();
    cout << "---------------------------------------------\n";
    return processes;
}

// Function to set power profile
void adjustPowerProfile(double totalCPUUsage) {
    string profile;
    if (totalCPUUsage < 20.0) {
        profile = "powersave";
    } else if (totalCPUUsage < 50.0) {
        profile = "balanced";
    } else {
        profile = "latency-performance";
    }

    string command = "bash -c 'tuned-adm profile " + profile + "'";
    cout << "Adjusting power profile to: " << profile << endl;

    int result = system(command.c_str());
    if (result != 0) {
        cerr << "Failed to set power profile. Check permissions or environment variables." << endl;
    }
}

// Function to kill a process by name
void killProcessByName(const string& processName) {
    string command = "pkill -f " + processName;
    cout << "Attempting to kill process: " << processName << endl;

    int result = system(command.c_str());
    if (result == 0) {
        cout << "Process " << processName << " terminated successfully." << endl;
    } else {
        cerr << "Failed to terminate process " << processName << ". It may not exist." << endl;
    }
}

// Function to ask user if they want to kill a process (with a timeout)
void askToKillProcess() {
    cout << "Do you want to kill a process? (yes/no): ";

    // Run input operation asynchronously with a timeout
    future<string> userInput = async(launch::async, [] {
        string response;
        cin >> response;
        return response;
    });

    // Wait for input with a timeout of 10 seconds
    if (userInput.wait_for(chrono::seconds(10)) == future_status::ready) {
        string response = userInput.get();
        if (response == "yes") {
            cout << "Enter the process name to kill: ";
            string processName;
            cin >> processName;
            killProcessByName(processName);
        }
    } else {
        cout << "No response in 10 seconds. Moving forward...\n";
    }
}

int main() {
    while (true) {
        cout << "Running csvExport to collect process data..." << endl;
        system("./csvExport");  

        this_thread::sleep_for(chrono::seconds(10));  // Wait for csvExport to complete

        cout << "Reading processes.csv to determine CPU usage..." << endl;
        vector<ProcessInfo> processes = getProcessDetails();

        // Calculate total CPU usage
        double totalCPUUsage = 0.0;
        for (const auto& process : processes) {
            totalCPUUsage += process.cpuUsage;
        }
        cout << "Total CPU Usage: " << totalCPUUsage << "%" << endl;

        adjustPowerProfile(totalCPUUsage);

        askToKillProcess();  // Ask user if they want to kill a process (with timeout)

        cout << "Sleeping for 2 minutes before next update...\n" << endl;
        this_thread::sleep_for(chrono::minutes(2)); // Wait 2 minutes
    }

    return 0;
}
