package com.ivan.learning.lesson2;

public class GradientHomework {
    private static final double STEP_SIZE = 0.3;
    private static final double POINT_INTERVAL = 1;
    private static final int MAX_ITERATIONS = 600;

    private static final int RANGE_X_MIN = 0;
    private static final int RANGE_X_MAX = 20;
    private static final int RANGE_Y_MIN = -200;
    private static final int RANGE_Y_MAX = 200;

    private static double func(double x, double y) {
        return (Math.pow(0.4f * x - 5.024f, 4) + Math.pow(0.3f * y + 1.884f, 2) + 0.6 * Math.sin(x * y));
    }

    private static double[] computeGradientStep(double currentX, double currentY) {
        double gradientX = (func(currentX + STEP_SIZE, currentY) -
                func(currentX - STEP_SIZE, currentY)) / (2 * STEP_SIZE);
        double gradientY = (func(currentX, currentY + STEP_SIZE) -
                func(currentX, currentY - STEP_SIZE)) / (2 * STEP_SIZE);
        return new double[] {currentX - gradientX, currentY - gradientY};
    }

    public static void main(String[] args) {
        double bestX = 0, bestY = 0;
        double bestInitialX = 0, bestInitialY = 0;
        double minValue = Double.MAX_VALUE;

        long totalPoints = (long) ((long)((RANGE_X_MAX - RANGE_X_MIN) / POINT_INTERVAL) *
                                ((RANGE_Y_MAX - RANGE_Y_MIN) / POINT_INTERVAL));
        long pointsProcessed = 0;
        double progressStep = totalPoints / 100.0;
        double nextProgressThreshold = progressStep;

        System.out.printf("Общее количество точек: %d%n", totalPoints);
        System.out.printf("Интервалы точек для x: [%d; %d]%n", RANGE_X_MIN, RANGE_X_MAX);
        System.out.printf("Интервалы точек для y: [%d; %d]%n", RANGE_Y_MIN, RANGE_Y_MAX);
        System.out.printf("Интервалы между точками: %f%n", POINT_INTERVAL);
        System.out.printf("Размер шага градиентного спуска: %f%n", STEP_SIZE);
        System.out.printf("Итераций градиентного спуска: %d%n", MAX_ITERATIONS);

        for (double initialX = RANGE_X_MIN; initialX < RANGE_X_MAX; initialX += POINT_INTERVAL) {
            for (double initialY = RANGE_Y_MIN; initialY < RANGE_Y_MAX; initialY += POINT_INTERVAL) {
                double x = initialX;
                double y = initialY;

                for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                    double[] nextPoint = computeGradientStep(x, y);
                    x = nextPoint[0];
                    y = nextPoint[1];
                }

                double currentValue = func(x, y);

                if (currentValue < minValue) {
                    bestInitialX = initialX;
                    bestInitialY = initialY;
                    bestX = x;
                    bestY = y;
                    minValue = currentValue;
                }

                pointsProcessed++;
                if (pointsProcessed >= nextProgressThreshold) {
                    double progress = (pointsProcessed * 100.0) / totalPoints;
                    System.out.printf("\rПрогресс: %.0f%%", progress);
                    System.out.flush();
                    nextProgressThreshold += progressStep;
                }
            }
        }

        System.out.println();

        System.out.printf("Минимальное значение: %.6f%n", minValue);
        System.out.printf("При x = %.6f, y = %.6f%n", bestX, bestY);
        System.out.printf("С начальными значениями x = %.6f, y = %.6f%n",
                bestInitialX, bestInitialY);
    }
}
