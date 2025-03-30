#include "HighCPUProcesses.h"
#include "ProcessInfo.h"
#include <iostream>
#include <fstream> 
#include <vector>
#include <iomanip>
using namespace std;

extern vector<ProcessInfo> highCPUProcesses;

void exportCSV(const string& filename){
    ofstream file(filename , ios::app);
    if(!file.is_open()){
        cerr<<"Error: Unable to open file"<<filename<<endl;
        return;
    }

    file.seekp(0, ios::end); 
    if (file.tellp() == 0) { 
        file << "Date,PID,ProcessName,MemoryUsage,CPU\n";
    }
    
    for(const auto& process: highCPUProcesses){
        file<< process.date<< ","
            << process.pid<<","
            << process.name<<","
            << process.memUsage<<","
            << fixed <<  setprecision(2) <<process.cpuUsage << "\n";

    }
    file.close();
    cout << "CSV exported" << filename << endl;

}

int main(){
    listHighCPUProcesses();
    
    exportCSV("processes.csv");

    
}