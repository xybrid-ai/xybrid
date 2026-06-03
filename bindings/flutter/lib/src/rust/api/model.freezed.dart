// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

/// @nodoc
mixin _$FfiLoadEvent {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(double field0) progress,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(double field0)? progress,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(double field0)? progress,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiLoadEvent_Progress value) progress,
    required TResult Function(FfiLoadEvent_Complete value) complete,
    required TResult Function(FfiLoadEvent_Error value) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiLoadEvent_Progress value)? progress,
    TResult? Function(FfiLoadEvent_Complete value)? complete,
    TResult? Function(FfiLoadEvent_Error value)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiLoadEvent_Progress value)? progress,
    TResult Function(FfiLoadEvent_Complete value)? complete,
    TResult Function(FfiLoadEvent_Error value)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiLoadEventCopyWith<$Res> {
  factory $FfiLoadEventCopyWith(
          FfiLoadEvent value, $Res Function(FfiLoadEvent) then) =
      _$FfiLoadEventCopyWithImpl<$Res, FfiLoadEvent>;
}

/// @nodoc
class _$FfiLoadEventCopyWithImpl<$Res, $Val extends FfiLoadEvent>
    implements $FfiLoadEventCopyWith<$Res> {
  _$FfiLoadEventCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiLoadEvent_ProgressImplCopyWith<$Res> {
  factory _$$FfiLoadEvent_ProgressImplCopyWith(
          _$FfiLoadEvent_ProgressImpl value,
          $Res Function(_$FfiLoadEvent_ProgressImpl) then) =
      __$$FfiLoadEvent_ProgressImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double field0});
}

/// @nodoc
class __$$FfiLoadEvent_ProgressImplCopyWithImpl<$Res>
    extends _$FfiLoadEventCopyWithImpl<$Res, _$FfiLoadEvent_ProgressImpl>
    implements _$$FfiLoadEvent_ProgressImplCopyWith<$Res> {
  __$$FfiLoadEvent_ProgressImplCopyWithImpl(_$FfiLoadEvent_ProgressImpl _value,
      $Res Function(_$FfiLoadEvent_ProgressImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiLoadEvent_ProgressImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as double,
    ));
  }
}

/// @nodoc

class _$FfiLoadEvent_ProgressImpl extends FfiLoadEvent_Progress {
  const _$FfiLoadEvent_ProgressImpl(this.field0) : super._();

  @override
  final double field0;

  @override
  String toString() {
    return 'FfiLoadEvent.progress(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiLoadEvent_ProgressImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiLoadEvent_ProgressImplCopyWith<_$FfiLoadEvent_ProgressImpl>
      get copyWith => __$$FfiLoadEvent_ProgressImplCopyWithImpl<
          _$FfiLoadEvent_ProgressImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(double field0) progress,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return progress(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(double field0)? progress,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return progress?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(double field0)? progress,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (progress != null) {
      return progress(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiLoadEvent_Progress value) progress,
    required TResult Function(FfiLoadEvent_Complete value) complete,
    required TResult Function(FfiLoadEvent_Error value) error,
  }) {
    return progress(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiLoadEvent_Progress value)? progress,
    TResult? Function(FfiLoadEvent_Complete value)? complete,
    TResult? Function(FfiLoadEvent_Error value)? error,
  }) {
    return progress?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiLoadEvent_Progress value)? progress,
    TResult Function(FfiLoadEvent_Complete value)? complete,
    TResult Function(FfiLoadEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (progress != null) {
      return progress(this);
    }
    return orElse();
  }
}

abstract class FfiLoadEvent_Progress extends FfiLoadEvent {
  const factory FfiLoadEvent_Progress(final double field0) =
      _$FfiLoadEvent_ProgressImpl;
  const FfiLoadEvent_Progress._() : super._();

  double get field0;

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiLoadEvent_ProgressImplCopyWith<_$FfiLoadEvent_ProgressImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiLoadEvent_CompleteImplCopyWith<$Res> {
  factory _$$FfiLoadEvent_CompleteImplCopyWith(
          _$FfiLoadEvent_CompleteImpl value,
          $Res Function(_$FfiLoadEvent_CompleteImpl) then) =
      __$$FfiLoadEvent_CompleteImplCopyWithImpl<$Res>;
}

/// @nodoc
class __$$FfiLoadEvent_CompleteImplCopyWithImpl<$Res>
    extends _$FfiLoadEventCopyWithImpl<$Res, _$FfiLoadEvent_CompleteImpl>
    implements _$$FfiLoadEvent_CompleteImplCopyWith<$Res> {
  __$$FfiLoadEvent_CompleteImplCopyWithImpl(_$FfiLoadEvent_CompleteImpl _value,
      $Res Function(_$FfiLoadEvent_CompleteImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc

class _$FfiLoadEvent_CompleteImpl extends FfiLoadEvent_Complete {
  const _$FfiLoadEvent_CompleteImpl() : super._();

  @override
  String toString() {
    return 'FfiLoadEvent.complete()';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiLoadEvent_CompleteImpl);
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(double field0) progress,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return complete();
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(double field0)? progress,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return complete?.call();
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(double field0)? progress,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete();
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiLoadEvent_Progress value) progress,
    required TResult Function(FfiLoadEvent_Complete value) complete,
    required TResult Function(FfiLoadEvent_Error value) error,
  }) {
    return complete(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiLoadEvent_Progress value)? progress,
    TResult? Function(FfiLoadEvent_Complete value)? complete,
    TResult? Function(FfiLoadEvent_Error value)? error,
  }) {
    return complete?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiLoadEvent_Progress value)? progress,
    TResult Function(FfiLoadEvent_Complete value)? complete,
    TResult Function(FfiLoadEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete(this);
    }
    return orElse();
  }
}

abstract class FfiLoadEvent_Complete extends FfiLoadEvent {
  const factory FfiLoadEvent_Complete() = _$FfiLoadEvent_CompleteImpl;
  const FfiLoadEvent_Complete._() : super._();
}

/// @nodoc
abstract class _$$FfiLoadEvent_ErrorImplCopyWith<$Res> {
  factory _$$FfiLoadEvent_ErrorImplCopyWith(_$FfiLoadEvent_ErrorImpl value,
          $Res Function(_$FfiLoadEvent_ErrorImpl) then) =
      __$$FfiLoadEvent_ErrorImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class __$$FfiLoadEvent_ErrorImplCopyWithImpl<$Res>
    extends _$FfiLoadEventCopyWithImpl<$Res, _$FfiLoadEvent_ErrorImpl>
    implements _$$FfiLoadEvent_ErrorImplCopyWith<$Res> {
  __$$FfiLoadEvent_ErrorImplCopyWithImpl(_$FfiLoadEvent_ErrorImpl _value,
      $Res Function(_$FfiLoadEvent_ErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiLoadEvent_ErrorImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiLoadEvent_ErrorImpl extends FfiLoadEvent_Error {
  const _$FfiLoadEvent_ErrorImpl(this.field0) : super._();

  @override
  final String field0;

  @override
  String toString() {
    return 'FfiLoadEvent.error(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiLoadEvent_ErrorImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiLoadEvent_ErrorImplCopyWith<_$FfiLoadEvent_ErrorImpl> get copyWith =>
      __$$FfiLoadEvent_ErrorImplCopyWithImpl<_$FfiLoadEvent_ErrorImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(double field0) progress,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return error(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(double field0)? progress,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return error?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(double field0)? progress,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiLoadEvent_Progress value) progress,
    required TResult Function(FfiLoadEvent_Complete value) complete,
    required TResult Function(FfiLoadEvent_Error value) error,
  }) {
    return error(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiLoadEvent_Progress value)? progress,
    TResult? Function(FfiLoadEvent_Complete value)? complete,
    TResult? Function(FfiLoadEvent_Error value)? error,
  }) {
    return error?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiLoadEvent_Progress value)? progress,
    TResult Function(FfiLoadEvent_Complete value)? complete,
    TResult Function(FfiLoadEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(this);
    }
    return orElse();
  }
}

abstract class FfiLoadEvent_Error extends FfiLoadEvent {
  const factory FfiLoadEvent_Error(final String field0) =
      _$FfiLoadEvent_ErrorImpl;
  const FfiLoadEvent_Error._() : super._();

  String get field0;

  /// Create a copy of FfiLoadEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiLoadEvent_ErrorImplCopyWith<_$FfiLoadEvent_ErrorImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiStreamEvent {
  Object get field0 => throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiStreamToken field0) token,
    required TResult Function(FfiResult field0) complete,
    required TResult Function(String field0) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamToken field0)? token,
    TResult? Function(FfiResult field0)? complete,
    TResult? Function(String field0)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiStreamToken field0)? token,
    TResult Function(FfiResult field0)? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiStreamEvent_Token value) token,
    required TResult Function(FfiStreamEvent_Complete value) complete,
    required TResult Function(FfiStreamEvent_Error value) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamEvent_Token value)? token,
    TResult? Function(FfiStreamEvent_Complete value)? complete,
    TResult? Function(FfiStreamEvent_Error value)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiStreamEvent_Token value)? token,
    TResult Function(FfiStreamEvent_Complete value)? complete,
    TResult Function(FfiStreamEvent_Error value)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiStreamEventCopyWith<$Res> {
  factory $FfiStreamEventCopyWith(
          FfiStreamEvent value, $Res Function(FfiStreamEvent) then) =
      _$FfiStreamEventCopyWithImpl<$Res, FfiStreamEvent>;
}

/// @nodoc
class _$FfiStreamEventCopyWithImpl<$Res, $Val extends FfiStreamEvent>
    implements $FfiStreamEventCopyWith<$Res> {
  _$FfiStreamEventCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiStreamEvent_TokenImplCopyWith<$Res> {
  factory _$$FfiStreamEvent_TokenImplCopyWith(_$FfiStreamEvent_TokenImpl value,
          $Res Function(_$FfiStreamEvent_TokenImpl) then) =
      __$$FfiStreamEvent_TokenImplCopyWithImpl<$Res>;
  @useResult
  $Res call({FfiStreamToken field0});
}

/// @nodoc
class __$$FfiStreamEvent_TokenImplCopyWithImpl<$Res>
    extends _$FfiStreamEventCopyWithImpl<$Res, _$FfiStreamEvent_TokenImpl>
    implements _$$FfiStreamEvent_TokenImplCopyWith<$Res> {
  __$$FfiStreamEvent_TokenImplCopyWithImpl(_$FfiStreamEvent_TokenImpl _value,
      $Res Function(_$FfiStreamEvent_TokenImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiStreamEvent_TokenImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as FfiStreamToken,
    ));
  }
}

/// @nodoc

class _$FfiStreamEvent_TokenImpl extends FfiStreamEvent_Token {
  const _$FfiStreamEvent_TokenImpl(this.field0) : super._();

  @override
  final FfiStreamToken field0;

  @override
  String toString() {
    return 'FfiStreamEvent.token(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiStreamEvent_TokenImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiStreamEvent_TokenImplCopyWith<_$FfiStreamEvent_TokenImpl>
      get copyWith =>
          __$$FfiStreamEvent_TokenImplCopyWithImpl<_$FfiStreamEvent_TokenImpl>(
              this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiStreamToken field0) token,
    required TResult Function(FfiResult field0) complete,
    required TResult Function(String field0) error,
  }) {
    return token(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamToken field0)? token,
    TResult? Function(FfiResult field0)? complete,
    TResult? Function(String field0)? error,
  }) {
    return token?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiStreamToken field0)? token,
    TResult Function(FfiResult field0)? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (token != null) {
      return token(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiStreamEvent_Token value) token,
    required TResult Function(FfiStreamEvent_Complete value) complete,
    required TResult Function(FfiStreamEvent_Error value) error,
  }) {
    return token(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamEvent_Token value)? token,
    TResult? Function(FfiStreamEvent_Complete value)? complete,
    TResult? Function(FfiStreamEvent_Error value)? error,
  }) {
    return token?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiStreamEvent_Token value)? token,
    TResult Function(FfiStreamEvent_Complete value)? complete,
    TResult Function(FfiStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (token != null) {
      return token(this);
    }
    return orElse();
  }
}

abstract class FfiStreamEvent_Token extends FfiStreamEvent {
  const factory FfiStreamEvent_Token(final FfiStreamToken field0) =
      _$FfiStreamEvent_TokenImpl;
  const FfiStreamEvent_Token._() : super._();

  @override
  FfiStreamToken get field0;

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiStreamEvent_TokenImplCopyWith<_$FfiStreamEvent_TokenImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiStreamEvent_CompleteImplCopyWith<$Res> {
  factory _$$FfiStreamEvent_CompleteImplCopyWith(
          _$FfiStreamEvent_CompleteImpl value,
          $Res Function(_$FfiStreamEvent_CompleteImpl) then) =
      __$$FfiStreamEvent_CompleteImplCopyWithImpl<$Res>;
  @useResult
  $Res call({FfiResult field0});
}

/// @nodoc
class __$$FfiStreamEvent_CompleteImplCopyWithImpl<$Res>
    extends _$FfiStreamEventCopyWithImpl<$Res, _$FfiStreamEvent_CompleteImpl>
    implements _$$FfiStreamEvent_CompleteImplCopyWith<$Res> {
  __$$FfiStreamEvent_CompleteImplCopyWithImpl(
      _$FfiStreamEvent_CompleteImpl _value,
      $Res Function(_$FfiStreamEvent_CompleteImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiStreamEvent_CompleteImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as FfiResult,
    ));
  }
}

/// @nodoc

class _$FfiStreamEvent_CompleteImpl extends FfiStreamEvent_Complete {
  const _$FfiStreamEvent_CompleteImpl(this.field0) : super._();

  @override
  final FfiResult field0;

  @override
  String toString() {
    return 'FfiStreamEvent.complete(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiStreamEvent_CompleteImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiStreamEvent_CompleteImplCopyWith<_$FfiStreamEvent_CompleteImpl>
      get copyWith => __$$FfiStreamEvent_CompleteImplCopyWithImpl<
          _$FfiStreamEvent_CompleteImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiStreamToken field0) token,
    required TResult Function(FfiResult field0) complete,
    required TResult Function(String field0) error,
  }) {
    return complete(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamToken field0)? token,
    TResult? Function(FfiResult field0)? complete,
    TResult? Function(String field0)? error,
  }) {
    return complete?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiStreamToken field0)? token,
    TResult Function(FfiResult field0)? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiStreamEvent_Token value) token,
    required TResult Function(FfiStreamEvent_Complete value) complete,
    required TResult Function(FfiStreamEvent_Error value) error,
  }) {
    return complete(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamEvent_Token value)? token,
    TResult? Function(FfiStreamEvent_Complete value)? complete,
    TResult? Function(FfiStreamEvent_Error value)? error,
  }) {
    return complete?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiStreamEvent_Token value)? token,
    TResult Function(FfiStreamEvent_Complete value)? complete,
    TResult Function(FfiStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete(this);
    }
    return orElse();
  }
}

abstract class FfiStreamEvent_Complete extends FfiStreamEvent {
  const factory FfiStreamEvent_Complete(final FfiResult field0) =
      _$FfiStreamEvent_CompleteImpl;
  const FfiStreamEvent_Complete._() : super._();

  @override
  FfiResult get field0;

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiStreamEvent_CompleteImplCopyWith<_$FfiStreamEvent_CompleteImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiStreamEvent_ErrorImplCopyWith<$Res> {
  factory _$$FfiStreamEvent_ErrorImplCopyWith(_$FfiStreamEvent_ErrorImpl value,
          $Res Function(_$FfiStreamEvent_ErrorImpl) then) =
      __$$FfiStreamEvent_ErrorImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class __$$FfiStreamEvent_ErrorImplCopyWithImpl<$Res>
    extends _$FfiStreamEventCopyWithImpl<$Res, _$FfiStreamEvent_ErrorImpl>
    implements _$$FfiStreamEvent_ErrorImplCopyWith<$Res> {
  __$$FfiStreamEvent_ErrorImplCopyWithImpl(_$FfiStreamEvent_ErrorImpl _value,
      $Res Function(_$FfiStreamEvent_ErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiStreamEvent_ErrorImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiStreamEvent_ErrorImpl extends FfiStreamEvent_Error {
  const _$FfiStreamEvent_ErrorImpl(this.field0) : super._();

  @override
  final String field0;

  @override
  String toString() {
    return 'FfiStreamEvent.error(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiStreamEvent_ErrorImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiStreamEvent_ErrorImplCopyWith<_$FfiStreamEvent_ErrorImpl>
      get copyWith =>
          __$$FfiStreamEvent_ErrorImplCopyWithImpl<_$FfiStreamEvent_ErrorImpl>(
              this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiStreamToken field0) token,
    required TResult Function(FfiResult field0) complete,
    required TResult Function(String field0) error,
  }) {
    return error(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamToken field0)? token,
    TResult? Function(FfiResult field0)? complete,
    TResult? Function(String field0)? error,
  }) {
    return error?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiStreamToken field0)? token,
    TResult Function(FfiResult field0)? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiStreamEvent_Token value) token,
    required TResult Function(FfiStreamEvent_Complete value) complete,
    required TResult Function(FfiStreamEvent_Error value) error,
  }) {
    return error(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiStreamEvent_Token value)? token,
    TResult? Function(FfiStreamEvent_Complete value)? complete,
    TResult? Function(FfiStreamEvent_Error value)? error,
  }) {
    return error?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiStreamEvent_Token value)? token,
    TResult Function(FfiStreamEvent_Complete value)? complete,
    TResult Function(FfiStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(this);
    }
    return orElse();
  }
}

abstract class FfiStreamEvent_Error extends FfiStreamEvent {
  const factory FfiStreamEvent_Error(final String field0) =
      _$FfiStreamEvent_ErrorImpl;
  const FfiStreamEvent_Error._() : super._();

  @override
  String get field0;

  /// Create a copy of FfiStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiStreamEvent_ErrorImplCopyWith<_$FfiStreamEvent_ErrorImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiTtsStreamEvent {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(Uint8List pcm, int sampleRate) audioChunk,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiTtsStreamEvent_AudioChunk value) audioChunk,
    required TResult Function(FfiTtsStreamEvent_Complete value) complete,
    required TResult Function(FfiTtsStreamEvent_Error value) error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult? Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult? Function(FfiTtsStreamEvent_Error value)? error,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult Function(FfiTtsStreamEvent_Error value)? error,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiTtsStreamEventCopyWith<$Res> {
  factory $FfiTtsStreamEventCopyWith(
          FfiTtsStreamEvent value, $Res Function(FfiTtsStreamEvent) then) =
      _$FfiTtsStreamEventCopyWithImpl<$Res, FfiTtsStreamEvent>;
}

/// @nodoc
class _$FfiTtsStreamEventCopyWithImpl<$Res, $Val extends FfiTtsStreamEvent>
    implements $FfiTtsStreamEventCopyWith<$Res> {
  _$FfiTtsStreamEventCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiTtsStreamEvent_AudioChunkImplCopyWith<$Res> {
  factory _$$FfiTtsStreamEvent_AudioChunkImplCopyWith(
          _$FfiTtsStreamEvent_AudioChunkImpl value,
          $Res Function(_$FfiTtsStreamEvent_AudioChunkImpl) then) =
      __$$FfiTtsStreamEvent_AudioChunkImplCopyWithImpl<$Res>;
  @useResult
  $Res call({Uint8List pcm, int sampleRate});
}

/// @nodoc
class __$$FfiTtsStreamEvent_AudioChunkImplCopyWithImpl<$Res>
    extends _$FfiTtsStreamEventCopyWithImpl<$Res,
        _$FfiTtsStreamEvent_AudioChunkImpl>
    implements _$$FfiTtsStreamEvent_AudioChunkImplCopyWith<$Res> {
  __$$FfiTtsStreamEvent_AudioChunkImplCopyWithImpl(
      _$FfiTtsStreamEvent_AudioChunkImpl _value,
      $Res Function(_$FfiTtsStreamEvent_AudioChunkImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? pcm = null,
    Object? sampleRate = null,
  }) {
    return _then(_$FfiTtsStreamEvent_AudioChunkImpl(
      pcm: null == pcm
          ? _value.pcm
          : pcm // ignore: cast_nullable_to_non_nullable
              as Uint8List,
      sampleRate: null == sampleRate
          ? _value.sampleRate
          : sampleRate // ignore: cast_nullable_to_non_nullable
              as int,
    ));
  }
}

/// @nodoc

class _$FfiTtsStreamEvent_AudioChunkImpl extends FfiTtsStreamEvent_AudioChunk {
  const _$FfiTtsStreamEvent_AudioChunkImpl(
      {required this.pcm, required this.sampleRate})
      : super._();

  @override
  final Uint8List pcm;
  @override
  final int sampleRate;

  @override
  String toString() {
    return 'FfiTtsStreamEvent.audioChunk(pcm: $pcm, sampleRate: $sampleRate)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiTtsStreamEvent_AudioChunkImpl &&
            const DeepCollectionEquality().equals(other.pcm, pcm) &&
            (identical(other.sampleRate, sampleRate) ||
                other.sampleRate == sampleRate));
  }

  @override
  int get hashCode => Object.hash(
      runtimeType, const DeepCollectionEquality().hash(pcm), sampleRate);

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiTtsStreamEvent_AudioChunkImplCopyWith<
          _$FfiTtsStreamEvent_AudioChunkImpl>
      get copyWith => __$$FfiTtsStreamEvent_AudioChunkImplCopyWithImpl<
          _$FfiTtsStreamEvent_AudioChunkImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(Uint8List pcm, int sampleRate) audioChunk,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return audioChunk(pcm, sampleRate);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return audioChunk?.call(pcm, sampleRate);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (audioChunk != null) {
      return audioChunk(pcm, sampleRate);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiTtsStreamEvent_AudioChunk value) audioChunk,
    required TResult Function(FfiTtsStreamEvent_Complete value) complete,
    required TResult Function(FfiTtsStreamEvent_Error value) error,
  }) {
    return audioChunk(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult? Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult? Function(FfiTtsStreamEvent_Error value)? error,
  }) {
    return audioChunk?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult Function(FfiTtsStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (audioChunk != null) {
      return audioChunk(this);
    }
    return orElse();
  }
}

abstract class FfiTtsStreamEvent_AudioChunk extends FfiTtsStreamEvent {
  const factory FfiTtsStreamEvent_AudioChunk(
      {required final Uint8List pcm,
      required final int sampleRate}) = _$FfiTtsStreamEvent_AudioChunkImpl;
  const FfiTtsStreamEvent_AudioChunk._() : super._();

  Uint8List get pcm;
  int get sampleRate;

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiTtsStreamEvent_AudioChunkImplCopyWith<
          _$FfiTtsStreamEvent_AudioChunkImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiTtsStreamEvent_CompleteImplCopyWith<$Res> {
  factory _$$FfiTtsStreamEvent_CompleteImplCopyWith(
          _$FfiTtsStreamEvent_CompleteImpl value,
          $Res Function(_$FfiTtsStreamEvent_CompleteImpl) then) =
      __$$FfiTtsStreamEvent_CompleteImplCopyWithImpl<$Res>;
}

/// @nodoc
class __$$FfiTtsStreamEvent_CompleteImplCopyWithImpl<$Res>
    extends _$FfiTtsStreamEventCopyWithImpl<$Res,
        _$FfiTtsStreamEvent_CompleteImpl>
    implements _$$FfiTtsStreamEvent_CompleteImplCopyWith<$Res> {
  __$$FfiTtsStreamEvent_CompleteImplCopyWithImpl(
      _$FfiTtsStreamEvent_CompleteImpl _value,
      $Res Function(_$FfiTtsStreamEvent_CompleteImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc

class _$FfiTtsStreamEvent_CompleteImpl extends FfiTtsStreamEvent_Complete {
  const _$FfiTtsStreamEvent_CompleteImpl() : super._();

  @override
  String toString() {
    return 'FfiTtsStreamEvent.complete()';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiTtsStreamEvent_CompleteImpl);
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(Uint8List pcm, int sampleRate) audioChunk,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return complete();
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return complete?.call();
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete();
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiTtsStreamEvent_AudioChunk value) audioChunk,
    required TResult Function(FfiTtsStreamEvent_Complete value) complete,
    required TResult Function(FfiTtsStreamEvent_Error value) error,
  }) {
    return complete(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult? Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult? Function(FfiTtsStreamEvent_Error value)? error,
  }) {
    return complete?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult Function(FfiTtsStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (complete != null) {
      return complete(this);
    }
    return orElse();
  }
}

abstract class FfiTtsStreamEvent_Complete extends FfiTtsStreamEvent {
  const factory FfiTtsStreamEvent_Complete() = _$FfiTtsStreamEvent_CompleteImpl;
  const FfiTtsStreamEvent_Complete._() : super._();
}

/// @nodoc
abstract class _$$FfiTtsStreamEvent_ErrorImplCopyWith<$Res> {
  factory _$$FfiTtsStreamEvent_ErrorImplCopyWith(
          _$FfiTtsStreamEvent_ErrorImpl value,
          $Res Function(_$FfiTtsStreamEvent_ErrorImpl) then) =
      __$$FfiTtsStreamEvent_ErrorImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class __$$FfiTtsStreamEvent_ErrorImplCopyWithImpl<$Res>
    extends _$FfiTtsStreamEventCopyWithImpl<$Res, _$FfiTtsStreamEvent_ErrorImpl>
    implements _$$FfiTtsStreamEvent_ErrorImplCopyWith<$Res> {
  __$$FfiTtsStreamEvent_ErrorImplCopyWithImpl(
      _$FfiTtsStreamEvent_ErrorImpl _value,
      $Res Function(_$FfiTtsStreamEvent_ErrorImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? field0 = null,
  }) {
    return _then(_$FfiTtsStreamEvent_ErrorImpl(
      null == field0
          ? _value.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiTtsStreamEvent_ErrorImpl extends FfiTtsStreamEvent_Error {
  const _$FfiTtsStreamEvent_ErrorImpl(this.field0) : super._();

  @override
  final String field0;

  @override
  String toString() {
    return 'FfiTtsStreamEvent.error(field0: $field0)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiTtsStreamEvent_ErrorImpl &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiTtsStreamEvent_ErrorImplCopyWith<_$FfiTtsStreamEvent_ErrorImpl>
      get copyWith => __$$FfiTtsStreamEvent_ErrorImplCopyWithImpl<
          _$FfiTtsStreamEvent_ErrorImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(Uint8List pcm, int sampleRate) audioChunk,
    required TResult Function() complete,
    required TResult Function(String field0) error,
  }) {
    return error(field0);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult? Function()? complete,
    TResult? Function(String field0)? error,
  }) {
    return error?.call(field0);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(Uint8List pcm, int sampleRate)? audioChunk,
    TResult Function()? complete,
    TResult Function(String field0)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(field0);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiTtsStreamEvent_AudioChunk value) audioChunk,
    required TResult Function(FfiTtsStreamEvent_Complete value) complete,
    required TResult Function(FfiTtsStreamEvent_Error value) error,
  }) {
    return error(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult? Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult? Function(FfiTtsStreamEvent_Error value)? error,
  }) {
    return error?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiTtsStreamEvent_AudioChunk value)? audioChunk,
    TResult Function(FfiTtsStreamEvent_Complete value)? complete,
    TResult Function(FfiTtsStreamEvent_Error value)? error,
    required TResult orElse(),
  }) {
    if (error != null) {
      return error(this);
    }
    return orElse();
  }
}

abstract class FfiTtsStreamEvent_Error extends FfiTtsStreamEvent {
  const factory FfiTtsStreamEvent_Error(final String field0) =
      _$FfiTtsStreamEvent_ErrorImpl;
  const FfiTtsStreamEvent_Error._() : super._();

  String get field0;

  /// Create a copy of FfiTtsStreamEvent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiTtsStreamEvent_ErrorImplCopyWith<_$FfiTtsStreamEvent_ErrorImpl>
      get copyWith => throw _privateConstructorUsedError;
}
