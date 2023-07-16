import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('star_data.csv')


data['Mass'] = data['Mass'] * 1.989e+30  
data['Radius'] = data['Radius'] * 6.957e+8 


gravity_values = []

def calculate_gravity(mass, radius, distance):
    G = 6.67430e-11  
    gravity = (G * mass) / (radius ** 2)
    return gravity

for index, row in data.iterrows():
    mass = row['Mass']
    radius = row['Radius']
    distance = row['Distance']
    gravity = calculate_gravity(mass, radius, distance)
    gravity_values.append(gravity)


data['Gravity'] = gravity_values


radius_data = data['Radius'].values.reshape(-1, 1)
mass_data = data['Mass'].values.reshape(-1, 1)


inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(radius_data)
    inertias.append(kmeans.inertia_)


plt.plot(range(1, 11), inertias)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(radius_data)

plt.scatter(radius_data, mass_data, c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')
plt.xlabel('Radius (m)')
plt.ylabel('Mass (kg)')
plt.title(f'Clusters (k={k})')
plt.grid(True)
plt.show()

