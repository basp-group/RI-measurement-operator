function flag = isclose(a, b, atol, rtol)

flag = all(abs(a(:) - b(:)) <= atol + rtol * abs(b(:)));
