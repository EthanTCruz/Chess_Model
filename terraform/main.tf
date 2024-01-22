terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.51.0"
    }
  }
}


provider "google" {
  credentials = file("C:/Users/ethan/git/Full_Chess_App/Chess_Model/terraform/secret.json")
  region      = var.region
  project = var.project_id
}


resource "google_container_cluster" "gke_cluster" {
  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count = 1

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "model"
  #may be making multiple nodes due  to thinking there are multiple zones and maybe location needs to be a zone
  location   = var.region
  cluster    = google_container_cluster.gke_cluster.name
  node_count = 1

  node_config {
    machine_type = var.machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    preemptible = true
  }
}


provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "gke_full-chess_us-central1_chess-app"
}


resource "kubernetes_pod" "high_resource_ml_pod" {
  metadata {
    name = "self-training-pod"
    labels = {
      app = "ml-model"
    }
  }

  spec {
    container {
      image = "ethancruz/chess_model:v1.0.4"  // Replace with your container image
      name  = "self-training"

      resources {
        requests = {
          memory = "8Gi"   // Requesting 8 GB of memory
          cpu    = "2000m" // Requesting 2 CPU cores (2000 milliCPU units)
        }
        limits = {
          memory = "16Gi"  // Setting a limit of 16 GB of memory
          cpu    = "4000m" // Setting a limit of 4 CPU cores
        }
      }
    }
  }
}
