from django.utils import timezone  
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.hashers import check_password
from .models import User
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.urls import reverse
from django.template.loader import render_to_string
from django.utils.html import strip_tags

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
        'updated_at': user.updated_at,
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
    if 'username' in request.data:
       username = request.data['username']
       if User.objects.filter(username=username).exclude(id=user.id).exists():
          return Response(
              {'error': 'Ce nom d’utilisateur est déjà utilisé'},
              status=status.HTTP_400_BAD_REQUEST
          )
       user.username = username

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




@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def change_password(request):
    """
    Changer le mot de passe si l'utilisateur connaît son mot de passe actuel
    """
    user = request.user
    current_password = request.data.get('current_password')
    new_password = request.data.get('new_password')
    confirm_password = request.data.get('confirm_password')

    if not check_password(current_password, user.password):
        return Response({'error': 'Mot de passe actuel incorrect'}, status=status.HTTP_400_BAD_REQUEST)
    
    if new_password != confirm_password:
        return Response({'error': 'Les mots de passe ne correspondent pas'}, status=status.HTTP_400_BAD_REQUEST)
    
    if len(new_password) < 8:
        return Response({'error': 'Le mot de passe doit contenir au moins 8 caractères'}, status=status.HTTP_400_BAD_REQUEST)
    
    user.set_password(new_password)
    user.save()
    
    update_session_auth_hash(request, user)

    return Response({'message': 'Mot de passe changé avec succès'})


@api_view(['PUT'])
@permission_classes([AllowAny])
def request_password_reset(request):
    email = request.data.get('email')

    if not email:
        return Response({'error': 'Email requis'}, status=400)

    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({'message': 'Si cet email existe, un lien a été envoyé.'})

    token = default_token_generator.make_token(user)
    uid = user.pk

    reset_url = f"http://localhost:3000/reset-password/{uid}/{token}"

    subject = "Réinitialisation de votre mot de passe"
    html_message = render_to_string("emails/reset_password.html", {"reset_url": reset_url, "user": user})
    plain_message = strip_tags(html_message)

    send_mail(
       subject,
       plain_message,
       "noreply@tonapp.com",
       [email],
       html_message=html_message,
    )

    return Response({'message': 'Si cet email existe, un lien a été envoyé.'})


@api_view(['PUT'])
@permission_classes([AllowAny])
def reset_password(request, uid, token):
    new_password = request.data.get('new_password')
    confirm_password = request.data.get('confirm_password')

    if not new_password or not confirm_password:
        return Response({'error': 'Tous les champs sont requis'}, status=400)

    if new_password != confirm_password:
        return Response({'error': 'Les mots de passe ne correspondent pas'}, status=400)

    if len(new_password) < 8:
        return Response({'error': 'Mot de passe trop court'}, status=400)

    try:
        user = User.objects.get(pk=uid)
    except User.DoesNotExist:
        return Response({'error': 'Utilisateur invalide'}, status=400)

    if not default_token_generator.check_token(user, token):
        return Response({'error': 'Token invalide ou expiré'}, status=400)

    user.set_password(new_password)
    user.save()

    return Response({'message': 'Mot de passe réinitialisé avec succès'})


from django.utils import timezone