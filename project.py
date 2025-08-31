import matplotlib.pyplot as plt

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Show few predictions
print("\nSample predictions:")
for i in range(5):
    print("Headline:", X_test[i], "-> Predicted Sentiment:", y_pred[i])

# Count distribution of predicted labels
labels_count = pd.Series(y_pred).value_counts()

# Print counts
print("\nSentiment distribution:")
print(labels_count)

# Plot Pie Chart
plt.figure(figsize=(6,6))
plt.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=90)
plt.title("Sentiment Distribution of News Headlines")
plt.show()
