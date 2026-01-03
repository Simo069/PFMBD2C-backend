from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from .models import User

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """
    Inscription d'un nouvel utilisateur
    """
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')
    full_name = request.data.get('full_name', '')
    
    # Validation
    if not username or not email or not password:
        return Response(
            {'error': 'Nom d\'utilisateur, email et mot de passe requis'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Vérifier si l'utilisateur existe
    if User.objects.filter(username=username).exists():
        return Response(
            {'error': 'Ce nom d\'utilisateur existe déjà'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if User.objects.filter(email=email).exists():
        return Response(
            {'error': 'Cet email existe déjà'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Créer l'utilisateur
    user = User.objects.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name
    )
    
    # Générer les tokens JWT
    refresh = RefreshToken.for_user(user)
    
    return Response({
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
        },
        'tokens': {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """
    Connexion d'un utilisateur
    """
    username = request.data.get('username')
    password = request.data.get('password')
    
    if not username or not password:
        return Response(
            {'error': 'Nom d\'utilisateur et mot de passe requis'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Authentifier l'utilisateur
    user = authenticate(username=username, password=password)
    
    if user is None:
        return Response(
            {'error': 'Identifiants invalides'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # Générer les tokens JWT
    refresh = RefreshToken.for_user(user)
    
    # Mettre à jour last_login
    user.last_login = timezone.now()
    user.save(update_fields=['last_login'])
    
    return Response({
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
        },
        'tokens': {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    """
    Récupérer le profil de l'utilisateur connecté
    """
    user = request.user
    
    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'full_name': user.full_name,
        'created_at': user.created_at,
        'preferences': {
            'chunk_size': user.chunk_size,
            'chunk_overlap': user.chunk_overlap,
            'retrieval_k': user.retrieval_k,
        }
    })


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """
    Mettre à jour le profil utilisateur
    """
    
    user = request.user
    
    # Mettre à jour les champs
    if 'full_name' in request.data:
        user.full_name = request.data['full_name']
    
    if 'email' in request.data:
        email = request.data['email']
        if User.objects.filter(email=email).exclude(id=user.id).exists():
            return Response(
                {'error': 'Cet email est déjà utilisé'},
                status=status.HTTP_400_BAD_REQUEST
            )
        user.email = email
    
    # Mettre à jour les préférences
    if 'chunk_size' in request.data:
        user.chunk_size = request.data['chunk_size']
    if 'chunk_overlap' in request.data:
        user.chunk_overlap = request.data['chunk_overlap']
    if 'retrieval_k' in request.data:
        user.retrieval_k = request.data['retrieval_k']
    
    user.save()
    
    return Response({
        'message': 'Profil mis à jour avec succès',
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
        }
    })


from django.utils import timezone