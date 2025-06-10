//
//  Synchronization.hpp
//


#ifndef synchronization_hpp
#define synchronization_hpp

#import <Foundation/Foundation.h>


// Bootstrap service helpers for IOSurface sharing
int bootstrap_register_wrapper(const char* serviceName, int port);
int bootstrap_look_up_wrapper(const char* serviceName, int* outPort);

// Mach semaphore wrappers
int create_mach_semaphore(int* outSemaphore, int initialValue);
int signal_mach_semaphore(int semaphore);
int wait_mach_semaphore(int semaphore);
int destroy_mach_semaphore(int semaphore);

// Shared memory wrappers
void* create_shared_memory(const char* name, size_t size, int* outFileDescriptor);
void* open_shared_memory(const char* name, size_t size);
int close_shared_memory(void* ptr, size_t size, int fileDescriptor);


#endif
