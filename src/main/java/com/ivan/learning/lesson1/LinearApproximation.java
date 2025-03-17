package com.ivan.learning.lesson1;

import java.util.Map;

public class LinearApproximation {
    private static final Map<Float, Float> functionValues = Map.of(
            -0.15f, 0.6151f,
            -0.10f, 0.6418f,
            -0.05f, 0.6678f,
            0.00f, 0.6931f,
            0.05f, 0.7178f,
            0.10f, 0.7419f,
            0.15f, 0.7654f);

    public static float[] leastSquaresLinear() {
        int n = functionValues.size();
        float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (Map.Entry<Float, Float> entry : functionValues.entrySet()) {
            float x = entry.getKey();
            float y = entry.getValue();
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        float a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        float b = (sumY - a * sumX) / n;
        return new float[]{a, b};
    }

    public static void main(String[] args) {
        float[] coeffs = leastSquaresLinear();
        System.out.printf("f(x) = %.4f * x + %.4f%n", coeffs[0], coeffs[1]);
    }
}
// y = 0.5008 * x + 0.6918
// https://www.wolframalpha.com/input?i=plot+0.5008x+%2B+0.6918
//