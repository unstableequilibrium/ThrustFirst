#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/detail/type_traits.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <limits.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>


#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#endif

using namespace std;
using namespace thrust;

struct values_vec
{
	float X;
	float Y;
	float Z;
	float Vr;
	float Vs;
	float Vt;
};

void FirstReductionStep(vector<values_vec> &hostData, thrust::host_vector<int> &hostSortID, thrust::host_vector<int> &hostKeyData,
	vector<values_vec> &intermediate, vector<int> &intermediateKey, bool isBinary,
	ofstream &finalOut, int &reducedRecs, bool isOptionA = false, string pathInter = "")
{
	ofstream intermediateOut;
	vector<int> dataIds, keyIds;
	if (isOptionA){
#ifdef _WIN32
		string fileName = pathInter + string("\\intermediate1");
#elif __APPLE__
		string fileName = pathInter + string("/intermediate1");
#endif
		if (isBinary){
			fileName += string(".bin");
			intermediateOut.open(fileName.c_str(), ios::binary);
		}
		else{
			fileName += string(".txt");
			intermediateOut.open(fileName.c_str());
		}
	}
	reducedRecs = 0;
	// checking in treshold values
	int N = hostData.size();
	for (int j = 0; j < N; j++){
		// Find number of records with same key
		int sameKeys = 1;
		while (j < N - 1){
			if ((hostKeyData[j] >> 3) == (hostKeyData[j + 1] >> 3))j++, sameKeys++;
			else break;
		}

		int curID;
		if (sameKeys == 1 && isOptionA){
			curID = hostSortID[j];
			if (!isBinary){
				finalOut << hostData[curID].X << " " << hostData[curID].Y << " " << hostData[curID].Z << " ";
				finalOut << hostKeyData[j] << " ";
				finalOut << hostData[curID].Vr << " " << hostData[curID].Vs << " " << hostData[curID].Vt << std::endl;
			}
			continue;
		}

		// calculate mean values of current same key records
		values_vec meanVals;
		meanVals.Vr = meanVals.Vs = meanVals.Vt = 0;
		for (int i = j - sameKeys + 1; i <= j; i++){
			curID = hostSortID[i];
			meanVals.Vr += hostData[curID].Vr;
			meanVals.Vs += hostData[curID].Vs;
			meanVals.Vt += hostData[curID].Vt;
		}
		meanVals.Vr /= sameKeys;
		meanVals.Vs /= sameKeys;
		meanVals.Vt /= sameKeys;

		// calculate standard deviations 
		values_vec stdVals;
		stdVals.Vr = stdVals.Vs = stdVals.Vt = 0;
		for (int i = j - sameKeys + 1; i <= j; i++){
			curID = hostSortID[i];
			stdVals.Vr += (hostData[curID].Vr - meanVals.Vr) * (hostData[curID].Vr - meanVals.Vr);
			stdVals.Vs += (hostData[curID].Vs - meanVals.Vs) * (hostData[curID].Vs - meanVals.Vs);
			stdVals.Vt += (hostData[curID].Vt - meanVals.Vt) * (hostData[curID].Vt - meanVals.Vt);
		}
		stdVals.Vr = sqrtf(stdVals.Vr / sameKeys);
		stdVals.Vs = sqrtf(stdVals.Vs / sameKeys);
		stdVals.Vt = sqrtf(stdVals.Vt / sameKeys);

		// checking treshold
		bool inTresh = true;
		for (int i = j - sameKeys + 1; i <= j; i++){
			curID = hostSortID[i];
			int z1 = (int) (fabsf(hostData[curID].Vr - meanVals.Vr) / stdVals.Vr + 0.5);
			int z2 = (int) (fabsf(hostData[curID].Vs - meanVals.Vs) / stdVals.Vs + 0.5);
			int z3 = (int) (fabsf(hostData[curID].Vt - meanVals.Vt) / stdVals.Vt + 0.5);
			if (z1 != 1 || z2 != 1 || z3 != 1){
				inTresh = false;
				break;
			}
		}
		if (inTresh){
			// write to intermediate data
			int oldKey = hostKeyData[j - sameKeys + 1];
			int curID = hostSortID[j - sameKeys + 1];
			meanVals.X = hostData[curID].X; meanVals.Y = hostData[curID].Y; meanVals.Z = hostData[curID].Z;
			intermediate.push_back(meanVals);
			intermediateKey.push_back(oldKey);

			reducedRecs += sameKeys;
			if (isOptionA){
				if (!isBinary){
					intermediateOut << intermediate.back().X << " " << intermediate.back().Y << " " << intermediate.back().Z << " ";
					intermediateOut << (intermediateKey.back() >> 3) << " ";
					intermediateOut << intermediate.back().Vr << " " << intermediate.back().Vs << " " << intermediate.back().Vt << " ";
					intermediateOut << intermediateKey.back() << endl;
				}
			}
		}
		else if (isOptionA){
			// write to final file
			if (!isBinary){
				ostringstream out;
				for (int i = j - sameKeys + 1; i <= j; i++){
					curID = hostSortID[i];
					values_vec tmp = hostData[curID];
					out << tmp.X << " " << tmp.Y << " " << tmp.Z << " ";
					out << hostKeyData[i] << " ";
					out << tmp.Vr << " " << tmp.Vs << " " << tmp.Vt << std::endl;
				}
				finalOut << out.str();
			}
			else{
				for (int i = j - sameKeys + 1; i <= j; i++){
					dataIds.push_back(hostSortID[i]);
					keyIds.push_back(hostKeyData[i]);
				}
			}
		}
	}

	if (isOptionA){
		// write to binary files
		if (isBinary){
			int finalSz = dataIds.size();
			int reduceSz = finalSz >> 3;
			char *Buffer = new char[28 * reduceSz];
			float *ptr = (float *) Buffer;
			int counter = 0;
			vector<int>::iterator itK = keyIds.begin();
			for (int i = 0; i < finalSz; i++){
				int curID = dataIds[i];
				values_vec tmp = hostData[curID];
				*(ptr++) = tmp.X; *(ptr++) = tmp.Y; *(ptr++) = tmp.Z;
				*(ptr++) = *((float *) (&(*(itK++))));
				*(ptr++) = tmp.Vr; *(ptr++) = tmp.Vs; *(ptr++) = tmp.Vt;
				counter++;
				if (counter % reduceSz == 0){
					finalOut.write(Buffer, 28 * reduceSz);
					ptr = (float *) Buffer;
				}
			}
			int remain = counter % reduceSz;
			if (remain)finalOut.write(Buffer, 28 * remain);

			delete [] Buffer;

			// Intermediate
			finalSz = intermediate.size();
			Buffer = new char[32 * finalSz];
			ptr = (float *) Buffer;
			for (int i = 0; i < finalSz; i++){
				values_vec tmp = intermediate[i];
				*(ptr++) = tmp.X; *(ptr++) = tmp.Y; *(ptr++) = tmp.Z;
				int newKey = intermediateKey[i] >> 3;
				*(ptr++) = *((float *) (&newKey));
				*(ptr++) = tmp.Vr; *(ptr++) = tmp.Vs; *(ptr++) = tmp.Vt;
				*(ptr++) = *((float *) (&intermediateKey[i]));
			}
			intermediateOut.write(Buffer, 32 * finalSz);
			delete [] Buffer;
		}
		intermediateOut.close();
	}
}

void NextReductionStep(int step, vector<values_vec> &hostData, thrust::host_vector<int> &hostKeyData,
	vector<values_vec> &intermediate, vector<int> &intermediateKey, bool isBinary,
	ofstream &finalOut, int &reducedRecs, bool isOptionA = false, string pathInter = "")
{
	reducedRecs = 0;
	int N = hostData.size();
	if (N == 1 && isOptionA){
		if (isBinary){
			finalOut << hostData[0].X << hostData[0].Y << hostData[0].Z;
			finalOut << hostKeyData[0] << hostData[0].Vr << hostData[0].Vs << hostData[0].Vt;
		}
		else {
			finalOut << hostData[0].X << " " << hostData[0].Y << " " << hostData[0].Z << " ";
			finalOut << hostKeyData[0] << " ";
			finalOut << hostData[0].Vr << " " << hostData[0].Vs << " " << hostData[0].Vt << std::endl;
		}
		return;
	}

	ofstream intermediateOut;
	vector<int> finalIds;
	if (isOptionA){
		ostringstream filename;
#ifdef _WIN32
		filename << pathInter << "\\intermediate" << (step + 1);
#elif __APPLE__
		filename << pathInter << "/intermediate" << (step + 1);
#endif
		if (isBinary){
			filename << ".bin";
			intermediateOut.open(filename.str().c_str(), ios::binary);
		}
		else{
			filename << ".txt";
			intermediateOut.open(filename.str().c_str());
		}
	}

	// checking in treshold values
	int shift = 3 * step + 3;
	for (int j = 0; j < N; j++){
		// Find number of records with same key
		int sameKeys = 1;
		while (j < N - 1){
			if ((hostKeyData[j] >> shift) == (hostKeyData[j + 1] >> shift))j++, sameKeys++;
			else break;
		}

		if (sameKeys == 1 && isOptionA){
			if (!isBinary){
				finalOut << hostData[j].X << " " << hostData[j].Y << " " << hostData[j].Z << " ";
				finalOut << hostKeyData[j] << " ";
				finalOut << hostData[j].Vr << " " << hostData[j].Vs << " " << hostData[j].Vt << std::endl;
			}
			else{
				finalIds.push_back(j);
			}
			continue;
		}

		// calculate mean values of current same key records
		values_vec meanVals;
		meanVals.Vr = meanVals.Vs = meanVals.Vt = 0;
		for (int i = j - sameKeys + 1; i <= j; i++){
			meanVals.Vr += hostData[i].Vr;
			meanVals.Vs += hostData[i].Vs;
			meanVals.Vt += hostData[i].Vt;
		}
		meanVals.Vr /= sameKeys;
		meanVals.Vs /= sameKeys;
		meanVals.Vt /= sameKeys;

		// calculate standard deviations 
		values_vec stdVals;
		stdVals.Vr = stdVals.Vs = stdVals.Vt = 0;
		for (int i = j - sameKeys + 1; i <= j; i++){
			stdVals.Vr += (hostData[i].Vr - meanVals.Vr) * (hostData[i].Vr - meanVals.Vr);
			stdVals.Vs += (hostData[i].Vs - meanVals.Vs) * (hostData[i].Vs - meanVals.Vs);
			stdVals.Vt += (hostData[i].Vt - meanVals.Vt) * (hostData[i].Vt - meanVals.Vt);
		}
		stdVals.Vr = sqrtf(stdVals.Vr / sameKeys);
		stdVals.Vs = sqrtf(stdVals.Vs / sameKeys);
		stdVals.Vt = sqrtf(stdVals.Vt / sameKeys);

		// checking treshold
		bool inTresh = true;
		for (int i = j - sameKeys + 1; i <= j; i++){
			int z1 = (int) (fabsf(hostData[i].Vr - meanVals.Vr) / stdVals.Vr + 0.5);
			int z2 = (int) (fabsf(hostData[i].Vs - meanVals.Vs) / stdVals.Vs + 0.5);
			int z3 = (int) (fabsf(hostData[i].Vt - meanVals.Vt) / stdVals.Vt + 0.5);
			if (z1 != 1 || z2 != 1 || z3 != 1){
				inTresh = false;
				break;
			}
		}

		if (inTresh){
			// write to intermediate data
			int oldKey = hostKeyData[j - sameKeys + 1];
			int curID = j - sameKeys + 1;
			meanVals.X = hostData[curID].X; meanVals.Y = hostData[curID].Y; meanVals.Z = hostData[curID].Z;
			intermediate.push_back(meanVals);
			intermediateKey.push_back(oldKey);
			reducedRecs += sameKeys;
			if (isOptionA){
				if (!isBinary){
					intermediateOut << intermediate.back().X << " " << intermediate.back().Y << " " << intermediate.back().Z << " ";
					intermediateOut << (intermediateKey.back() >> 3) << " ";
					intermediateOut << intermediate.back().Vr << " " << intermediate.back().Vs << " " << intermediate.back().Vt << " ";
					intermediateOut << intermediateKey.back() << endl;
				}
			}
		}
		else if (isOptionA){
			if (!isBinary){
				ostringstream out;
				for (int i = j - sameKeys + 1; i <= j; i++){
					values_vec tmp = hostData[i];
					out << tmp.X << " " << tmp.Y << " " << tmp.Z << " ";
					out << hostKeyData[i] << " ";
					out << tmp.Vr << " " << tmp.Vs << " " << tmp.Vt << std::endl;
				}
				finalOut << out.str();
			}
			else{
				for (int i = j - sameKeys + 1; i <= j; i++){
					finalIds.push_back(i);
				}
			}
		}
	}
	if (isOptionA){
		// write to binary files
		if (isBinary){
			// Final
			int finalSz = finalIds.size();
			char *Buffer = new char[28 * finalSz];
			float *ptr = (float *) Buffer;
			for (int i = 0; i < finalSz; i++){
				int curID = finalIds[i];
				values_vec tmp = hostData[curID];
				*(ptr++) = tmp.X; *(ptr++) = tmp.Y; *(ptr++) = tmp.Z;
				*(ptr++) = *((float *) (&hostKeyData[curID]));
				*(ptr++) = tmp.Vr; *(ptr++) = tmp.Vs; *(ptr++) = tmp.Vt;
			}
			finalOut.write(Buffer, 28 * finalSz);
			delete [] Buffer;

			// Intermediate
			finalSz = intermediate.size();
			Buffer = new char[32 * finalSz];
			ptr = (float *) Buffer;
			for (int i = 0; i < finalSz; i++){
				values_vec tmp = intermediate[i];
				*(ptr++) = tmp.X; *(ptr++) = tmp.Y; *(ptr++) = tmp.Z;
				int newKey = intermediateKey[i] >> 3;
				*(ptr++) = *((float *) (&newKey));
				*(ptr++) = tmp.Vr; *(ptr++) = tmp.Vs; *(ptr++) = tmp.Vt;
				*(ptr++) = *((float *) (&intermediateKey[i]));
			}
			intermediateOut.write(Buffer, 32 * finalSz);
			delete [] Buffer;

		}
		intermediateOut.close();
	}
}

void Merge2SortedHalves(thrust::host_vector<int> &key, thrust::host_vector<int> &val, int begin, int end,
	thrust::host_vector<int> &keySorted, thrust::host_vector<int> &valSorted, int beginSorted)
{
	int j = 0, k = 0;
	int N = end - begin;
	int N2 = N >> 1;
	for (int i = N2; i < N; i++){
		if (j != N2){
			while (key[begin + i] > key[begin + j]){
				valSorted[beginSorted + k] = val[begin + j];
				keySorted[beginSorted + k] = key[begin + j];
				k++, j++;
				if (j == N2)break;
			}
		}
		valSorted[beginSorted + k] = val[begin + i];
		keySorted[beginSorted + k] = key[begin + i];
		k++;
	}
	while (j < N2){
		valSorted[beginSorted + k] = val[begin + j];
		keySorted[beginSorted + k] = key[begin + j];
		k++, j++;
	}
}

void OptimizedReadFile(int N, ifstream &input, char delim, vector<values_vec> &hostData, thrust::host_vector<int> &hostKeyData)
{
	char buf[256];
	vector<values_vec>::iterator it = hostData.begin();
	thrust::host_vector<int>::iterator itK = hostKeyData.begin();
	for (int i = 0; i < N; i++){
		input.getline(buf, 256);
		char *ch = buf;
		(*it).X = atof(ch);
		while (*ch++ != delim);
		(*it).Y = atof(ch);
		while (*ch++ != delim);
		(*it).Z = atof(ch);
		while (*ch++ != delim);
		while (*ch++ != delim);
		while (*ch++ != delim);
		while (*ch++ != delim);
		*itK = atoi(ch);
		while (*ch++ != delim);
		(*it).Vr = atof(ch);
		while (*ch++ != delim);
		(*it).Vs = atof(ch);
		while (*ch++ != delim);
		(*it).Vt = atof(ch);
		it++;
		itK++;
	}
	input.close();
}

void ReductionOfFile(int x, ifstream &input, float &TotalTime, bool isOptionA, bool withGPU, bool isBinary, ofstream &finalOut, ofstream &timeOut, string pathInter = "")
{
	int N = 1 << (3 * x);
	int N2 = N >> 1, N4 = N >> 2;
	clock_t cpu_time = clock();
	float allocationTime, readTime;
	// allocate host data of values on CPU
	std::vector<values_vec> hostData(N);
	thrust::host_vector<int> hostKeyData(N);
	thrust::host_vector<int> hostSortID(N);

	float totalSort = 0;
	if (withGPU){
		// additional memory for merging
		thrust::host_vector<int> hostKeyData1(N);
		thrust::host_vector<int> hostSortID1(N);

		// allocate thrust::device data on GPU
		thrust::device_vector<int> devKeyData(N2);
		thrust::device_vector<int> devSortID(N2);
		allocationTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "allocation Data time = " << allocationTime << std::endl;
		if (isOptionA)timeOut << "allocation Data time = " << allocationTime << std::endl;

		// Reading original file 
		//std::cout<<"start reading data from input to host \n";
		cpu_time = clock();

		// Optimized read original file which have N records with delimiter = '\t' 
		OptimizedReadFile(N, input, '\t', hostData, hostKeyData);
		// init vec of IDs
		for (int i = 0; i < N; i++)hostSortID[i] = i;

		readTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "reading file time = " << readTime << std::endl;
		if (isOptionA)timeOut << "reading file time = " << readTime << std::endl;

		// copy 1-st part host data to device GPU data 
		//std::cout<<"copy key data to GPU \n";
		cpu_time = clock();
		thrust::copy(hostKeyData.begin(), hostKeyData.begin() + N2, devKeyData.begin());
		thrust::copy(hostSortID.begin(), hostSortID.begin() + N2, devSortID.begin());
		float copyToGPU = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "copy keys to GPU time = " << copyToGPU << std::endl;
		if (isOptionA)timeOut << "copy keys to GPU time = " << copyToGPU << std::endl;

		// sort 1-st part key column device data
		//std::cout<<"sort key data on GPU \n";
		cpu_time = clock();
		thrust::sort_by_key(devKeyData.begin(), devKeyData.begin() + N4, devSortID.begin());
		thrust::sort_by_key(devKeyData.begin() + N4, devKeyData.begin() + N2, devSortID.begin() + N4);
		float sortTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "sort keys on GPU time = " << sortTime << std::endl;
		if (isOptionA)timeOut << "sort keys on GPU time = " << sortTime << std::endl;

		// copy data fron device to host
		//std::cout<<"copy key data from GPU to host \n";
		cpu_time = clock();
		thrust::copy(devKeyData.begin(), devKeyData.end(), hostKeyData.begin());
		thrust::copy(devSortID.begin(), devSortID.end(), hostSortID.begin());
		float copyToCPU = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "copy keys from GPU to CPU time = " << copyToCPU << std::endl;
		if (isOptionA)timeOut << "copy keys from GPU to CPU time = " << copyToCPU << std::endl;

		// copy 2-nd part host data to device GPU data 
		//std::cout<<"copy key data to GPU \n";
		cpu_time = clock();
		thrust::copy(hostKeyData.begin() + N2, hostKeyData.end(), devKeyData.begin());
		thrust::copy(hostSortID.begin() + N2, hostSortID.end(), devSortID.begin());
		float copyToGPU2 = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "copy keys to GPU time = " << copyToGPU << std::endl;
		if (isOptionA)timeOut << "copy keys to GPU time = " << copyToGPU << std::endl;

		// sort 2-nd part key column device data
		//std::cout<<"sort key data on GPU \n";
		cpu_time = clock();
		thrust::sort_by_key(devKeyData.begin(), devKeyData.begin() + N4, devSortID.begin());
		thrust::sort_by_key(devKeyData.begin() + N4, devKeyData.begin() + N2, devSortID.begin() + N4);
		float sortTime2 = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "sort keys on GPU time = " << sortTime << std::endl;
		if (isOptionA)timeOut << "sort keys on GPU time = " << sortTime << std::endl;

		// copy data fron device to host
		//std::cout<<"copy key data from GPU to host \n";
		cpu_time = clock();
		thrust::copy(devKeyData.begin(), devKeyData.end(), hostKeyData.begin() + N2);
		thrust::copy(devSortID.begin(), devSortID.end(), hostSortID.begin() + N2);
		float copyToCPU2 = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "copy keys from GPU to CPU time = " << copyToCPU << std::endl;
		if (isOptionA)timeOut << "copy keys from GPU to CPU time = " << copyToCPU << std::endl;

		// 2 Merging 4 sorted quarters from GPU into 2 sorted halves vector on CPU
		cpu_time = clock();
		Merge2SortedHalves(hostKeyData, hostSortID, 0, N2, hostKeyData1, hostSortID1, 0);
		Merge2SortedHalves(hostKeyData, hostSortID, N2, N, hostKeyData1, hostSortID1, N2);
		// Merging 2 halves
		Merge2SortedHalves(hostKeyData1, hostSortID1, 0, N, hostKeyData, hostSortID, 0);
		float mergeTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "Merge on CPU time = " << mergeTime << std::endl;
		if (isOptionA)timeOut << "Merge on CPU time = " << mergeTime << std::endl;

		totalSort += copyToCPU + copyToGPU + sortTime + copyToCPU2 + copyToGPU2 + sortTime2 + mergeTime;
		// clear unnecessary data
		hostKeyData1.clear();
		hostSortID1.clear();
	}
	else{
		allocationTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "allocation Data time = " << allocationTime << std::endl;
		if (isOptionA)timeOut << "allocation Data time = " << allocationTime << std::endl;

		// Reading original file 
		//std::cout<<"start reading data from input to host \n";
		cpu_time = clock();

		// Optimized read original file which have N records with delimiter = '\t' 
		OptimizedReadFile(N, input, '\t', hostData, hostKeyData);
		// init vec of IDs
		for (int i = 0; i < N; i++)hostSortID[i] = i;

		readTime = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "reading file time = " << readTime << std::endl;
		if (isOptionA)timeOut << "reading file time = " << readTime << std::endl;

		// sort keys 
		cpu_time = clock();
		thrust::sort_by_key(hostKeyData.begin(), hostKeyData.end(), hostSortID.begin());
		totalSort = float(clock() - cpu_time) / CLOCKS_PER_SEC;
		cout << "Sort on CPU time = " << totalSort << std::endl;
	}

	// First reduction step
	//std::cout<<"start 1-st reduction step \n";
	vector<values_vec> intermediate;
	vector<int> intermediateKey;
	vector<float> iterationTime;
	int reducedRecs = 0, finalRecs = 0;
	cpu_time = clock();
	FirstReductionStep(hostData, hostSortID, hostKeyData, intermediate, intermediateKey, isBinary, finalOut, reducedRecs, isOptionA, pathInter);
	iterationTime.push_back(float(clock() - cpu_time) / CLOCKS_PER_SEC);
	finalRecs += (hostKeyData.size() - reducedRecs);
	cout << "1-iteration time = " << iterationTime.back() << std::endl;
	cout << "After 1 iteration: " << intermediate.size() << " records to Intermediate step" << endl;
	cout << "                   " << (hostKeyData.size() - reducedRecs) << " records to Final output" << endl;
	if (isOptionA)timeOut << "1-iteration time = " << iterationTime.back() << std::endl;
	hostSortID.clear();

	// next intermediate reduction steps
	int k = 1;
	while (k < x && !intermediate.empty()){
		cpu_time = clock();
		hostData = intermediate;
		hostKeyData = intermediateKey;
		intermediate.clear();
		intermediateKey.clear();
		reducedRecs = 0;
		NextReductionStep(k, hostData, hostKeyData, intermediate, intermediateKey, isBinary, finalOut, reducedRecs, isOptionA, pathInter);
		iterationTime.push_back(float(clock() - cpu_time) / CLOCKS_PER_SEC);
		finalRecs += (hostKeyData.size() - reducedRecs);
		cout << (k + 1) << "-iteration time = " << iterationTime.back() << std::endl;
		cout << "After " << (k + 1) << "iteration: " << intermediate.size() << " records to Intermediate step" << endl;
		cout << "                   " << (hostKeyData.size() - reducedRecs) << " records to Final output" << endl;
		if (isOptionA)timeOut << (k + 1) << "-iteration time = " << iterationTime.back() << std::endl;
		k++;
	}
	cout << "Total Final records: " << finalRecs << endl;

	// Total iterations time
	float totalItTime = 0;
	for (int i = 0; i < iterationTime.size(); i++)totalItTime += iterationTime[i];
	// Total time
	TotalTime = allocationTime + readTime + totalSort + totalItTime;

	if (isOptionA){
		cout << "Total iterations time = " << totalItTime << std::endl;
		timeOut << "Total iterations time = " << totalItTime << std::endl;


		cout << "Total time = " << TotalTime << std::endl;
		timeOut << "Total time = " << TotalTime << std::endl;

		timeOut.close();
		finalOut.close();
	}
}

int main(int argc, char **argv)
{
	int x = 7;// degree of 8: N = 8^x
	int NumOfFiles = 1;// number of input files in directory or number of iteration for one file
	char pathToRoot[1024];// path to folder with input files
	ofstream totalTimeOut("totalTime.txt");
	float totalTime = 0;

	bool isOptionA = true;// if false will be Option B (only total time for same N files)
	bool withGPU = true;// if false without using GPU only CPU
	bool isBinary = true;// if false -> txt format
	// for option B file name of one file which will be iterated N times
	string inputSingleFile("input1.txt");

#ifdef _WIN32 // FOR Windows

	std::ofstream finalOut;
	std::ofstream timeOut;
	if (isOptionA){
		GetCurrentDirectory(1024, pathToRoot);
		string folderFinal = string(pathToRoot) + string("\\FINAL");
		string folderInter = string(pathToRoot) + string("\\INTERMEDIATE");
		string folderTime = string(pathToRoot) + string("\\TIME");
		CreateDirectory(folderFinal.c_str(), NULL);
		CreateDirectory(folderInter.c_str(), NULL);
		CreateDirectory(folderTime.c_str(), NULL);
		for (int i = 0; i < NumOfFiles; i++){
			ostringstream fileName;
			fileName << "input" << (i + 1) << ".txt";
			std::ifstream input(fileName.str().c_str());
			if (!input.is_open()){
				cout << "File " << fileName.str().c_str() << " not found." << endl;
				continue;
			}
			else {
				cout << "File " << fileName.str().c_str() << " is opened" << endl;
			}
			ostringstream finalFile;
			finalFile << folderFinal << "\\final" << (i + 1);
			if (isBinary){
				finalFile << ".bin";
				finalOut.open(finalFile.str().c_str(), ofstream::binary);
			}
			else {
				finalFile << ".txt";
				finalOut.open(finalFile.str().c_str());
			}
			ostringstream  timeFile;
			timeFile << folderTime << "\\time" << (i + 1) << ".txt";
			timeOut.open(timeFile.str().c_str());

			ostringstream folderName;
			folderName << folderInter << "\\" << (i + 1);
			CreateDirectory(folderName.str().c_str(), NULL);

			float time;
			ReductionOfFile(x, input, time, isOptionA, withGPU, isBinary, finalOut, timeOut, folderName.str());
			totalTime += time;
		}
	}
	else {// Option B
		for (int i = 0; i < NumOfFiles; i++){
			float time;
			std::ifstream input(inputSingleFile.c_str());
			if (!input.is_open()){
				cout << "File " << inputSingleFile.c_str() << " not found" << endl;
				continue;
			}
			else{
				cout << "File " << inputSingleFile.c_str() << " is opened" << endl;
			}
			ReductionOfFile(x, input, time, isOptionA, withGPU, isBinary, finalOut, timeOut);
			totalTime += time;
		}
	}

#elif __APPLE__ // FOR MAC OSX

	std::ofstream finalOut;
	std::ofstream timeOut;
	if (isOptionA){
		getcwd(pathToRoot, 1024);
		string folderFinal = string(pathToRoot) + string("/FINAL");
		string folderInter = string(pathToRoot) + string("/INTERMEDIATE");
		string folderTime = string(pathToRoot) + string("/TIME");
		mkdir(folderFinal.c_str(), 0777);
		mkdir(folderInter.c_str(), 0777);
		mkdir(folderTime.c_str(), 0777);
		for (int i = 0; i < NumOfFiles; i++){
			ostringstream fileName;
			fileName << "input" << (i + 1) << ".txt";
			std::ifstream input(fileName.str().c_str());
			if (!input.is_open()){
				cout << "File " << fileName.str().c_str() << " not found." << endl;
				continue;
			}
			else {
				cout << "File " << fileName.str().c_str() << " is opened" << endl;
			}
			ostringstream finalFile;
			finalFile << folderFinal << "/final" << (i + 1);
			if (isBinary){
				finalFile << ".bin";
				finalOut.open(finalFile.str().c_str(), , ofstream::binary);
			}
			else {
				finalFile << ".txt";
				finalOut.open(finalFile.str().c_str());
			}

			ostringstream  timeFile;
			timeFile << folderTime << "/time" << (i + 1) << ".txt";
			timeOut.open(timeFile.str().c_str());

			ostringstream folderName;
			folderName << folderInter << "/" << (i + 1);
			mkdir(folderName.str().c_str(), 0777);
			float time;
			ReductionOfFile(x, input, time, isOptionA, withGPU, isBinary, finalOut, timeOut, folderName.str());
			totalTime += time;
		}
	}
	else {// Option B
		for (int i = 0; i < NumOfFiles; i++){
			float time;
			std::ifstream input(inputSingleFile.c_str());
			if (!input.is_open()){
				cout << "File " << inputSingleFile.c_str() << " not found" << endl;
				continue;
			}
			else{
				cout << "File " << inputSingleFile.c_str() << " is opened" << endl;
			}
			ReductionOfFile(x, input, time, isOptionA, withGPU, isBinary, finalOut, timeOut);
			totalTime += time;
		}
	}


#endif

	cout << "Total time = " << totalTime << endl;
	totalTimeOut << totalTime;
	totalTimeOut.close();

	return 0;
}