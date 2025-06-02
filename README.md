# RSA
Attempts to solve the RSA factorization problem with some unique approaches to the problem:

factor.py: Presents 4 factorization algorithms that are guaranteed to find prime factors p and q when given n (which is the product of these two prime factors p and q). Takes advantage of the fact that if m is the midpoint between p and q, then n = pq = (m-r)(m+r) = m^2 - r^2 where p = m-r and q = m+r. In other words, for any n that is the product of two distinct primes (and p≠2 and q≠2), n can be written as the difference of two square numbers, and knowing these square numbers enables us to easily determine p and q.

factor_ml.py: Attempts to predict the distance between p and q given only n using machine learning. Method already has shockingly good success (can predict the distance between p and q to within some epsilon about 25% of the time). Project still undergoing research to determine true capabilities of this method.

PrimeFractions.py: class for computing decimal period of 1 / n (i.e. the length of the sequence of repeating digits produced by 1 / n). Theoretical work by me and my teammates have shown that the decimal period of 1 / n can be used to determine the factors of n. This aspect of the project is still undergoing heavy testing.