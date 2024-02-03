terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "5.14.0"
    }
  }
}


provider "google" {
  credentials = file("C:/Users/ethan/git/Full_Chess_App/Chess_Model/terraform/secret.json")
  region      = var.region
  project = var.project_id
}


resource "google_service_account" "my_service_account" {
  account_id   = "chess-model-svc-acct"
  display_name = "Chess Model bucket actor"
}
resource "google_service_account_key" "my_service_account_key" {
  service_account_id = google_service_account.my_service_account.name
}
resource "kubernetes_secret" "my_gcp_secret" {
  metadata {
    name = "my-gcp-secret"
  }

  data = {
    "key.json" = base64encode(google_service_account_key.my_service_account_key.private_key)
  }
}


resource "google_container_cluster" "gke_cluster" {
  name     = var.cluster_name
  location = var.region

  remove_default_node_pool = true
  initial_node_count = 1
  deletion_protection = false
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
      image = "ethancruz/chess_model:v1.0.8"  // Replace with your container image
      name  = "self-training"

      resources {
        requests = {
          memory = "24Gi"   // Requesting 8 GB of memory
          cpu    = "3000m" // Requesting 2 CPU cores (2000 milliCPU units)
        }
        limits = {
          memory = "30Gi"  // Setting a limit of 16 GB of memory
          cpu    = "4000m" // Setting a limit of 4 CPU cores
        }
      }
      volume_mount {
        mount_path = "/var/secrets/google"
        name       = "gcp-key"
        read_only  = true
      }
    }
        volume {
      name = "gcp-key"

      secret {
        secret_name = kubernetes_secret.my_gcp_secret.metadata[0].name
      }
    }
  }


}

resource "google_storage_bucket" "my_bucket" {
  name     = var.bucket_name
  location = var.bucket_location
  storage_class = var.storage_class
  force_destroy = true
}

