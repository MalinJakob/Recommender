import pandas as pd
import numpy as np
from lenskit.algorithms import basic, Recommender, user_knn
from lenskit import batch
import matplotlib.pyplot as plt

def main():
    """Main function to run the recommender system"""
    # Read data into a DataFrame and name columns according to Lenskit's standard
    # Read in data
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')

    # Rename columns to match what Lenskit expects
    ratings = ratings.rename(columns={
        'userId': 'user',
        'movieId': 'item',
        'rating': 'rating',
        'timestamp': 'timestamp'
    })

    movies = movies.rename(columns={
        'movieId': 'item'
    })

    def simulate_user_interactions(recommendations, interaction_prob=0.2):
        """Simulate user interactions with recommendations"""
        new_ratings = []
        timestamp = pd.Timestamp.now().timestamp()
        for _, row in recommendations.iterrows():
            if np.random.random() < interaction_prob:
                rating = np.random.choice([3.5, 4.0, 4.5, 5.0], p=[0.2, 0.3, 0.3, 0.2])
                new_ratings.append({
                    'user': row['user'],
                    'item': row['item'],
                    'rating': rating,
                    'timestamp': timestamp
                })
                timestamp += 1
        return pd.DataFrame(new_ratings)

    def temporal_diversity(recommendations, movies):
        """Calculate temporal diversity"""
        thesemovies = movies.copy()
        thesemovies['year'] = thesemovies['title'].str.extract(r'\((\d{4})\)')
        thesemovies['year'] = pd.to_numeric(thesemovies['year'])
        movie_years = thesemovies.set_index('item')['year']
        rec_years = recommendations['item'].map(movie_years)
        return rec_years.std()

    def item_catalog_coverage(recommendations, movies):
        """Calculate what percentage of all items are recommended at least once"""
        unique_recommended_items = recommendations['item'].nunique()
        total_items = len(movies)
        return unique_recommended_items / total_items

    def novelty(recommendations, ratings):
        """Calculate novelty value based on item popularity"""
        item_popularity = ratings.groupby('item').size()
        total_interactions = len(ratings)
        novelty_scores = recommendations['item'].map(
            lambda x: -np.log2(item_popularity.get(x, 1) / total_interactions)
        )
        return novelty_scores.mean()

    def mean_interactions(recommendations, ratings):
        """Calculate average number of interactions for recommended items"""
        item_interactions = ratings.groupby('item').size()
        recommended_items_interactions = recommendations['item'].map(item_interactions)
        return recommended_items_interactions.mean()

    # Initialize recommendation algorithms
    recommenders = {}
    recommenders['random'] = basic.Random()
    recommenders['most_popular'] = basic.MostPopular()
    recommenders['personal_topn'] = Recommender.adapt(basic.TopN(basic.Bias()))
    recommenders['personal_knn'] = Recommender.adapt(basic.KNN(basic.Bias()))

    # Parameters
    n_iterations = 10
    n_users = 100
    n_items = 20

    # Metrics for each iteration
    metrics_history = []

    # Run iterations
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Train algorithms with current ratings
        print("Training recommendations...")
        for recommender in recommenders:
            recommenders[recommender].fit(ratings)
        
        # Select random users for this iteration
        selected_users = np.random.choice(ratings['user'].unique(), size=n_users, replace=False)

        # Get recommendations for each algorithm
        print("Getting recommendations...")
        recommendations = {}
        for recommender in recommenders:
            recommendations[recommender] = batch.recommend(recommenders[recommender], selected_users, n_items)

        # Metrics
        available_metrics = ['novelty', 'item_coverage']

        # Calculate metrics for each algorithm
        print("Calculating metrics...")
        iteration_metrics = {}
        for recs in recommendations:
            metrics = {}
            metrics[f'novelty_{recs}'] = novelty(recommendations[recs], ratings)
            metrics[f'item_coverage_{recs}'] = item_catalog_coverage(recommendations[recs], movies)
            iteration_metrics.update(metrics)

        # Save metrics for this iteration
        metrics_history.append(iteration_metrics)

        # Simulate user interactions and update ratings
        print("Simulating user interactions...")
        for recs in recommendations:
            new_ratings = simulate_user_interactions(recommendations[recs])
            ratings = pd.concat([ratings, new_ratings], ignore_index=True)

    # Convert metrics history to DataFrame for easier analysis
    metrics_df = pd.DataFrame(metrics_history)

    # Print final summary
    print("\nCalculated metrics (mean over all iterations):")
    mean_metrics = metrics_df.mean()

    for metric in available_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for algo in recommenders.keys():
            value = mean_metrics[f'{metric}_{algo}']
            print(f"{algo.title()}: {value:.3f}")

    # Draw a graph for each metric
    for metric in available_metrics:
        plt.figure(figsize=(7, 3))
        plt.title(f'Development of {metric.replace("_", " ").title()} over iterations', fontsize=10)

        # Plot lines for each algorithm
        for algo in recommenders.keys():
            column_name = f'{metric}_{algo}'
            plt.plot(metrics_df.index, metrics_df[column_name], marker='o', label=algo.title())

        # Adjust plot
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()