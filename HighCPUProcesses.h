#ifndef HIGH_CPU_PROCESSES_H
#define HIGH_CPU_PROCESSES_H

#include <vector>
#include <mutex>
#include "ProcessInfo.h"

using namespace std;

// Externally declared variables (defined in one .cpp file)
extern vector<ProcessInfo> highCPUProcesses;
extern mutex dataMutex;

// Function to export processes
void exportHighCPUProcesses(const string &filename);

#endif
