//
//  Synchronization.mm
//

#include "Synchronization.hpp"

#import <Foundation/Foundation.h>
#include <unistd.h>

// Bootstrap service helpers for IOSurface sharing
#include <bootstrap.h>

int bootstrap_register_wrapper(const char* serviceName, int port) {
    // Deprecated, but I haven't successfully sent a mach port through a mach message yet,
    // and I haven't successfully set up an XPC service between two non-XPC processes.
    return bootstrap_register(bootstrap_port, const_cast<char*>(serviceName), port);
}

int bootstrap_look_up_wrapper(const char* serviceName, int* outPort) {
    mach_port_t port = 0;
    kern_return_t result = bootstrap_look_up(bootstrap_port, const_cast<char*>(serviceName), &port);
    if (result == KERN_SUCCESS) {
        *outPort = port;
        return 0;
    }
    return result;
}

// Mach semaphore wrappers
#include <mach/mach.h>
#include <mach/semaphore.h>
#include <mach/task.h>

int create_mach_semaphore(int* outSemaphore, int initialValue) {
    semaphore_t semaphore;
    kern_return_t result = semaphore_create(mach_task_self(), &semaphore, SYNC_POLICY_FIFO, initialValue);
    if (result == KERN_SUCCESS) {
        *outSemaphore = (int)semaphore;
        return 0;
    }
    return result;
}

int signal_mach_semaphore(int semaphore) {
    kern_return_t result = semaphore_signal((semaphore_t)semaphore);
    return (result == KERN_SUCCESS) ? 0 : result;
}

int wait_mach_semaphore(int semaphore) {
    kern_return_t result = semaphore_wait((semaphore_t)semaphore);
    return (result == KERN_SUCCESS) ? 0 : result;
}

int destroy_mach_semaphore(int semaphore) {
    kern_return_t result = semaphore_destroy(mach_task_self(), (semaphore_t)semaphore);
    return (result == KERN_SUCCESS) ? 0 : result;
}

// Shared memory wrappers
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <sys/unistd.h>

void* create_shared_memory(const char* name, size_t size, int* outFileDescriptor) {
    // Create shared memory object
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        return nullptr;
    }

    // Set size
    if (ftruncate(fd, size) == -1) {
        close(fd);
        shm_unlink(name);
        return nullptr;
    }

    // Map memory
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        shm_unlink(name);
        return nullptr;
    }

    *outFileDescriptor = fd;
    return ptr;
}

void* open_shared_memory(const char* name, size_t size) {
    // Open existing shared memory object
    int fd = shm_open(name, O_RDWR, 0666);
    if (fd == -1) {
        return nullptr;
    }

    // Map memory
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); // Can close fd after mapping

    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    return ptr;
}

int close_shared_memory(void* ptr, size_t size, int fileDescriptor) {
    int result = 0;

    // Unmap memory
    if (munmap(ptr, size) == -1) {
        result = -1;
    }

    // Close file descriptor
    if (close(fileDescriptor) == -1) {
        result = -1;
    }

    return result;
}
