#include "HighCPUProcesses.h"
#include "ProcessInfo.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

extern vector<ProcessInfo> highCPUProcesses;

void exportCSV(const string& filename) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }

    // Check if file is empty (newly created), then write headers
    file.seekp(0, ios::end);
    if (file.tellp() == 0) { 
        file << "Date,PID,ProcessName,MemoryUsage,CPU\n";
    }

    for (const auto& process : highCPUProcesses) {
        file << process.date << ","
             << process.pid << ","
             << process.name << ","
             << process.memUsage << ","
             << fixed << setprecision(2) << process.cpuUsage << "\n";
    }

    file.close();
    cout << "Data appended to: " << filename << endl;
}

void ensureFileExists(const string& filename) {
    ifstream checkFile(filename);
    if (!checkFile.good()) {  // If file does not exist, create it
        ofstream newFile(filename);
        if (newFile.is_open()) {
            cout << "Created file: " << filename << endl;
            newFile.close();
        } else {
            cerr << "Error: Unable to create file " << filename << endl;
        }
    }
}

int main() {
    listHighCPUProcesses();

    // Ensure "collected_processes.csv" exists before appending data
    ensureFileExists("collected_processes.csv");

    // Export data
    exportCSV("processes.csv");
    exportCSV("collected_processes.csv");

    return 0;
}
