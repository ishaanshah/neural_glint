#pragma once
/**
 * \brief Check if two ranges overlap
 * \param a1
 *      Start of first range
 * \param a2
 *      End of first range
 * \param b1
 *      Start of second range
 * \param b2
 *      End of second range
*/
template <typename T>
__device__ bool range_intersect(T a1, T a2, T b1, T b2) {
    return (a1 >= b1 && a1 <= b2) ||
           (a2 >= b1 && a2 <= b2) ||
           (b1 >= a1 && b1 <= a2) ||
           (b2 >= a1 && b2 <= a2);
}